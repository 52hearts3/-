import torch
import torch.nn as nn
import torchaudio
import random
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import glob

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNetAudio(nn.Module):
    def __init__(self, in_channels, out_channels, num_noise_types=14):
        super(UNetAudio, self).__init__()
        self.num_noise_types = num_noise_types

        # 编码器
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        # 瓶颈层
        self.bottleneck = ConvBlock(512, 1024)

        # 解码器
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # 输出层（人声掩码）
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # 噪声分类分支
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.noise_classifier = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_noise_types, kernel_size=1),
            nn.Sigmoid()  # 输出每种噪声类型的概率
        )

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def crop_or_pad(self, target, reference):
        """
        裁剪或填充目标张量以匹配参考张量的空间维度。
        """
        target_size = target.size()[2:]  # [height, width]
        ref_size = reference.size()[2:]  # [height, width]

        delta_h = ref_size[0] - target_size[0]
        delta_w = ref_size[1] - target_size[1]

        if delta_h == 0 and delta_w == 0:
            return target

        if delta_h > 0 or delta_w > 0:
            pad_top = delta_h // 2
            pad_bottom = delta_h - pad_top
            pad_left = delta_w // 2
            pad_right = delta_w - pad_left
            target = torch.nn.functional.pad(
                target, (pad_left, pad_right, pad_top, pad_bottom)
            )

        if delta_h < 0 or delta_w < 0:
            start_h = abs(delta_h) // 2
            start_w = abs(delta_w) // 2
            target = target[..., start_h:start_h + ref_size[0], start_w:start_w + ref_size[1]]

        return target

    def forward(self, x):
        # 存储输入尺寸以对齐输出
        input_size = x.size()[2:]

        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # 瓶颈层
        b = self.bottleneck(self.pool(e4))

        # 噪声分类分支
        noise_probs = self.global_pool(b)
        noise_probs = self.noise_classifier(noise_probs)
        noise_probs = noise_probs.squeeze(-1).squeeze(-1)  # [batch_size, num_noise_types]

        # 解码器，带跳跃连接
        d4 = self.upconv4(b)
        e4 = self.crop_or_pad(e4, d4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        e3 = self.crop_or_pad(e3, d3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        e2 = self.crop_or_pad(e2, d2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        e1 = self.crop_or_pad(e1, d1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # 输出人声掩码
        out = self.out_conv(d1)
        out = self.sigmoid(out)

        # 对齐输出尺寸与输入
        out = self.crop_or_pad(out, torch.zeros(1, 1, *input_size, device=out.device))

        return out, noise_probs

def wav_to_spectrogram(wav, n_fft=2048, hop_length=512, use_complex=False):
    """
    将 WAV 转换为谱图（幅度或实部+虚部）。
    输入：wav (batch_size, channels, samples)，channels=1 或 2
    输出：谱图 (batch_size, channels 或 channels*2, freq_bins, time_bins)，相位
    """
    window = torch.hann_window(n_fft, device=wav.device)

    transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_length, power=None, window_fn=lambda x: window
    )
    spec = transform(wav)  # [batch_size, channels, freq_bins, time_bins]

    if use_complex:
        real = spec.real
        imag = spec.imag
        spectrogram = torch.cat([real, imag], dim=1)
    else:
        spectrogram = spec.abs()

    phase = spec.angle()
    return spectrogram, phase

def spectrogram_to_wav(spectrogram, phase, n_fft=2048, hop_length=512, use_complex=False):
    """
    从谱图（幅度或实部+虚部）重构 WAV。
    输入：谱图 (batch_size, channels 或 channels*2, freq_bins, time_bins)，相位
    输出：wav (batch_size, channels, samples)
    """
    window = torch.hann_window(n_fft, device=spectrogram.device)

    # 调试：打印输入张量形状
    # print(f"Spectrogram shape: {spectrogram.shape}")
    # print(f"Phase shape: {phase.shape}")

    if use_complex:
        channels = spectrogram.shape[1] // 2
        real = spectrogram[:, :channels, :, :]
        imag = spectrogram[:, channels:, :, :]
        spec = torch.complex(real, imag)
    else:
        spec = spectrogram * torch.exp(1j * phase)

    # 确保 spec 形状为 [batch_size, channels, freq_bins, time_bins]
    batch_size, channels, freq_bins, time_bins = spec.shape
    # print(f"Spec shape before istft: {spec.shape}")

    # 重塑为 [batch_size * channels, freq_bins, time_bins] 以满足 istft 要求
    spec = spec.view(batch_size * channels, freq_bins, time_bins)
    # print(f"Spec shape after reshape: {spec.shape}")

    # 使用 istft 重构
    wav = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=time_bins * hop_length,  # 近似原始长度
        center=True,
        return_complex=False
    )

    # 调试：打印 istft 输出形状
    # print(f"Wav shape after istft: {wav.shape}")

    # 恢复形状为 [batch_size, channels, samples]
    samples = wav.shape[-1]
    wav = wav.view(batch_size, channels, samples)
    # print(f"Final wav shape: {wav.shape}")

    return wav

def separate_vocals(model, wav, n_fft=2048, hop_length=512, use_complex=False, device='cuda'):
    """
    使用 U-Net 模型从输入 WAV 中分离人声，并预测噪声类型概率。

    输入：
        model: UNetAudio 模型
        wav (torch.Tensor): 输入音频，形状 [batch_size, channels, samples]
        n_fft (int): STFT 的 FFT 点数
        hop_length (int): STFT 的跳跃长度
        use_complex (bool): 是否使用复数谱图
        device (str): 设备（'cuda' 或 'cpu'）

    输出：
        tuple: (vocals, noise_probs)
            - vocals (torch.Tensor): 分离的人声，形状 [batch_size, channels, samples]
            - noise_probs (torch.Tensor): 噪声类型概率，形状 [batch_size, num_noise_types]
    """
    model.eval()
    wav = wav.to(device)

    # 转换为谱图
    spectrogram, phase = wav_to_spectrogram(wav, n_fft, hop_length, use_complex)
    spectrogram = spectrogram.to(device)
    phase = phase.to(device)

    # 使用 U-Net 获取掩码和噪声概率
    with torch.no_grad():
        mask, noise_probs = model(spectrogram)

    # 应用掩码获取人声谱图
    vocal_spectrogram = spectrogram * mask

    # 重构 WAV
    vocals = spectrogram_to_wav(vocal_spectrogram, phase, n_fft, hop_length, use_complex)
    return vocals, noise_probs

class VocalAugmentor:
    """
    人声音频增强类，支持随机应用多种噪声增强，并返回增强音频及噪声类型标签。
    """
    def __init__(self, sample_rate=44100, p=0.5, max_augmentations=8, noise_folder=r'E:\美赛\2025年第十五届MathorCup数学应用挑战赛赛题\C题\增强音效'):
        self.sample_rate = sample_rate
        self.p = p
        self.max_augmentations = max_augmentations
        self.noise_folder = noise_folder

        # 定义增强函数及其名称
        self.augmentation_map = [
            ('background_noise', self._add_background_noise),
            ('impulse_noise', self._add_impulse_noise),
            ('band_noise', self._add_band_noise),
            ('piano_noise', self._add_piano_noise),
            ('siren_noise', self._add_siren_noise),
            ('hiss_noise', self._add_hiss_noise),
            ('ship_horn_noise', self._add_ship_horn_noise),
            ('electric_buzz_noise', self._add_electric_buzz_noise),
            ('wind_noise', self._add_wind_noise),
            ('crowd_noise', self._add_crowd_noise),
            ('mechanical_noise', self._add_mechanical_noise),
            ('water_flow_noise', self._add_water_flow_noise),
            ('violin_noise', self._add_violin_noise),
            ('external_audio_noise', self._add_external_audio_noise),
        ]
        self.noise_types = [name for name, _ in self.augmentation_map]

        # 加载外部音效文件
        self.noise_files = []
        if self.noise_folder and os.path.isdir(self.noise_folder):
            self.noise_files = glob.glob(os.path.join(self.noise_folder, "*.wav")) + \
                              glob.glob(os.path.join(self.noise_folder, "*.mp3"))
            if not self.noise_files:
                print(f"警告：噪声文件夹 {self.noise_folder} 中未找到 WAV 或 MP3 文件")
            else:
                print(f"找到 {len(self.noise_files)} 个音效文件：{self.noise_folder}")

    def __call__(self, wav):
        """
        对输入音频应用增强，并返回增强音频及噪声类型标签。

        参数：
            wav (torch.Tensor): 输入音频张量，形状为 [batch_size, channels, samples]

        返回：
            tuple: (augmented_wav, applied_augs)
                - augmented_wav (torch.Tensor): 增强后的音频
                - applied_augs (torch.Tensor): 噪声类型标签，形状为 [num_noise_types]
        """
        wav = wav.clone()
        batch_size, channels, samples = wav.shape
        device = wav.device

        # 随机选择增强数量
        num_augs = random.randint(1, self.max_augmentations)
        available_augs = [(name, aug) for name, aug in self.augmentation_map
                         if aug != self._add_external_audio_noise or self.noise_files]
        selected_augs = random.sample(available_augs, min(num_augs, len(available_augs)))

        # 记录应用的增强类型
        applied_augs = torch.zeros(len(self.noise_types), dtype=torch.float32, device=device)
        for name, aug in selected_augs:
            wav = aug(wav, samples, channels, device)
            idx = self.noise_types.index(name)
            applied_augs[idx] = 1.0

        # 确保输出长度一致
        if wav.shape[-1] != samples:
            wav = wav[..., :samples]
        elif wav.shape[-1] < samples:
            wav = torch.nn.functional.pad(wav, (0, samples - wav.shape[-1]))

        # 归一化
        wav = torch.clamp(wav, -1.0, 1.0)
        return wav, applied_augs

    def _add_background_noise(self, wav, samples, channels, device, snr_db=15):
        duration = samples / self.sample_rate
        min_total_duration = 0.8 * duration
        num_bursts = random.randint(2, 3)
        burst_duration = int((min_total_duration / num_bursts) * self.sample_rate)
        noise = torch.zeros(wav.shape[0], channels, samples, device=device)
        burst_starts = random.sample(range(samples - burst_duration), num_bursts)

        noise_type = random.choice(['white', 'pink'])
        for start in burst_starts:
            end = min(start + burst_duration, samples)
            if noise_type == 'white':
                burst_noise = torch.randn(wav.shape[0], channels, end - start, device=device)
            else:
                white = torch.randn(wav.shape[0], channels, (end - start) * 2, device=device)
                fft = torch.fft.rfft(white, dim=-1)
                freqs = torch.linspace(1, (end - start) // 2 + 1, fft.shape[-1], device=device)
                fft = fft / torch.sqrt(freqs)
                burst_noise = torch.fft.irfft(fft, n=(end - start) * 2, dim=-1)[..., :end - start]
            noise[:, :, start:end] += burst_noise

        signal_power = torch.mean(wav ** 2)
        noise_power = torch.mean(noise ** 2)
        if noise_power > 0:
            scale = torch.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
            noise = noise * scale

        return wav + noise

    def _add_impulse_noise(self, wav, samples, channels, device, intensity=0.15):
        duration = samples / self.sample_rate
        min_total_duration = 0.8 * duration
        num_bursts = random.randint(2, 3)
        burst_duration = int((min_total_duration / num_bursts) * self.sample_rate)
        noise = torch.zeros(wav.shape[0], channels, samples, device=device)
        burst_starts = random.sample(range(samples - burst_duration), num_bursts)

        for start in burst_starts:
            end = min(start + burst_duration, samples)
            num_impulses = int((end - start) * intensity / self.sample_rate) * 2
            impulse_locs = random.sample(range(start, end), min(num_impulses, end - start))
            for loc in impulse_locs:
                amplitude = random.uniform(0.15, 0.6) * random.choice([-1, 1])
                noise[..., loc] += amplitude

        return wav + noise

    def _add_band_noise(self, wav, samples, channels, device, bandwidth=1500):
        duration = samples / self.sample_rate
        min_total_duration = 0.8 * duration
        num_bursts = random.randint(2, 3)
        burst_duration = int((min_total_duration / num_bursts) * self.sample_rate)
        noise = torch.zeros(wav.shape[0], channels, samples, device=device)
        burst_starts = random.sample(range(samples - burst_duration), num_bursts)

        for start in burst_starts:
            end = min(start + burst_duration, samples)
            t = torch.linspace(0, (end - start) / self.sample_rate, end - start, device=device)
            center_freq = random.uniform(500, self.sample_rate / 4)
            burst_noise = torch.randn(wav.shape[0], channels, end - start, device=device)
            fft = torch.fft.rfft(burst_noise, dim=-1)
            freqs = torch.linspace(0, self.sample_rate / 2, fft.shape[-1], device=device)
            mask = (freqs >= center_freq - bandwidth / 2) & (freqs <= center_freq + bandwidth / 2)
            fft = fft * mask.float()
            burst_noise = torch.fft.irfft(fft, n=end - start, dim=-1)
            noise[:, :, start:end] += burst_noise

        signal_power = torch.mean(wav ** 2)
        noise_power = torch.mean(noise ** 2)
        if noise_power > 0:
            scale = torch.sqrt(signal_power / noise_power) * 0.15
            noise = noise * scale

        return wav + noise

    def _add_piano_noise(self, wav, samples, channels, device, snr_db=12):
        duration = samples / self.sample_rate
        min_total_duration = 0.8 * duration
        num_bursts = random.randint(2, 3)
        burst_durations = [random.uniform(2, 6) for _ in range(num_bursts)]
        total_burst_duration = sum(burst_durations)
        if total_burst_duration < min_total_duration:
            scale = min_total_duration / total_burst_duration
            burst_durations = [d * scale for d in burst_durations]
        burst_samples = [int(d * self.sample_rate) for d in burst_durations]
        t = torch.linspace(0, duration, samples, device=device)
        noise = torch.zeros(wav.shape[0], channels, samples, device=device)
        available_starts = list(range(samples - max(burst_samples)))
        burst_starts = []
        for burst_duration in burst_samples:
            if available_starts:
                start = random.choice(available_starts)
                burst_starts.append(start)
                start_min = max(0, start - burst_duration)
                start_max = min(samples, start + burst_duration * 2)
                available_starts = [s for s in available_starts if s < start_min or s > start_max]

        major_progressions = [
            [[0, 4, 7], [5, 9, 12], [7, 11, 14], [0, 4, 7]],
            [[0, 4, 7], [9, 12, 16], [5, 9, 12], [0, 4, 7]],
            [[7, 11, 14], [0, 4, 7], [5, 9, 12], [7, 11, 14]],
        ]
        minor_progressions = [
            [[0, 3, 7], [5, 8, 12], [7, 10, 14], [0, 3, 7]],
            [[9, 12, 16], [0, 3, 7], [5, 8, 12], [9, 12, 16]],
            [[7, 10, 14], [0, 3, 7], [9, 12, 16], [7, 10, 14]],
        ]

        for start, burst_duration in zip(burst_starts, burst_samples):
            end = min(start + burst_duration, samples)
            num_chords = random.randint(4, 8)
            chord_duration = int((end - start) / num_chords)
            progression_type = random.choice(['major', 'minor'])
            progression = major_progressions if progression_type == 'major' else minor_progressions
            chord_sequence = random.choice(progression)

            for i in range(num_chords):
                chord_start = start + i * chord_duration
                chord_end = min(chord_start + chord_duration, end)
                chord = chord_sequence[i % len(chord_sequence)]
                base_freq = 440 * 2 ** (random.randint(-12, 12) / 12)

                segment = torch.zeros_like(t[chord_start:chord_end])
                t_segment = t[chord_start:chord_end]
                for note in chord:
                    freq = base_freq * 2 ** (note / 12)
                    note_wave = (
                        torch.sin(2 * np.pi * freq * t_segment) +
                        0.5 * torch.sin(2 * np.pi * (freq * 2) * t_segment) +
                        0.3 * torch.sin(2 * np.pi * (freq * 3) * t_segment)
                    )
                    envelope = torch.exp(-t_segment * 2.0) * (1 - torch.exp(-t_segment * 10.0))
                    segment += note_wave * envelope

                if i % 4 == 3 and random.random() < 0.3:
                    arpeggio_duration = int(chord_duration / len(chord))
                    for j, note in enumerate(chord):
                        note_start = chord_start + j * arpeggio_duration
                        note_end = min(note_start + arpeggio_duration, chord_end)
                        freq = base_freq * 2 ** (note / 12)
                        arp_segment = (
                            torch.sin(2 * np.pi * freq * t[note_start:note_end]) +
                            0.5 * torch.sin(2 * np.pi * (freq * 2) * t[note_start:note_end]) +
                            0.3 * torch.sin(2 * np.pi * (freq * 3) * t[note_start:note_end])
                        )
                        arp_envelope = torch.exp(-t[note_start:note_end] * 2.0) * (
                            1 - torch.exp(-t[note_start:note_end] * 10.0))
                        segment[note_start - chord_start:note_end - chord_start] += arp_segment * arp_envelope

                for c in range(channels):
                    noise[:, c, chord_start:chord_end] += segment

            print(f"Burst start: {start}, end: {end}, chord_sequence: {chord_sequence}")

        signal_power = torch.mean(wav ** 2)
        noise_power = torch.mean(noise ** 2)
        if noise_power > 0:
            scale = torch.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
            noise = noise * scale

        return wav + noise

    def _add_siren_noise(self, wav, samples, channels, device, snr_db=12):
        duration = samples / self.sample_rate
        min_total_duration = 0.8 * duration
        num_bursts = random.randint(2, 3)
        burst_duration = int((min_total_duration / num_bursts) * self.sample_rate)
        noise = torch.zeros(wav.shape[0], channels, samples, device=device)
        burst_starts = random.sample(range(samples - burst_duration), num_bursts)

        for start in burst_starts:
            end = min(start + burst_duration, samples)
            t = torch.linspace(0, (end - start) / self.sample_rate, end - start, device=device)
            freq = 500 + 500 * torch.sin(2 * np.pi * 0.8 * t)
            burst_noise = torch.sin(2 * np.pi * freq * t).reshape(1, 1, end - start).repeat(wav.shape[0], channels, 1)
            noise[:, :, start:end] += burst_noise.squeeze(-1)

        signal_power = torch.mean(wav ** 2)
        noise_power = torch.mean(noise ** 2)
        if noise_power > 0:
            scale = torch.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
            noise = noise * scale

        return wav + noise

    def _add_hiss_noise(self, wav, samples, channels, device, snr_db=12):
        duration = samples / self.sample_rate
        min_total_duration = 0.8 * duration
        num_bursts = random.randint(2, 3)
        burst_duration = int((min_total_duration / num_bursts) * self.sample_rate)
        noise = torch.zeros(wav.shape[0], channels, samples, device=device)
        burst_starts = random.sample(range(samples - burst_duration), num_bursts)

        for start in burst_starts:
            end = min(start + burst_duration, samples)
            t = torch.linspace(0, (end - start) / self.sample_rate, end - start, device=device)
            center_freq = random.uniform(4000, 8000)
            bandwidth = 800
            burst_noise = torch.randn(wav.shape[0], channels, end - start, device=device)
            fft = torch.fft.rfft(burst_noise, dim=-1)
            freqs = torch.linspace(0, self.sample_rate / 2, fft.shape[-1], device=device)
            mask = (freqs >= center_freq - bandwidth / 2) & (freqs <= center_freq + bandwidth / 2)
            fft = fft * mask.float()
            burst_noise = torch.fft.irfft(fft, n=end - start, dim=-1)
            noise[:, :, start:end] += burst_noise

        signal_power = torch.mean(wav ** 2)
        noise_power = torch.mean(noise ** 2)
        if noise_power > 0:
            scale = torch.sqrt(signal_power / noise_power) * 0.12
            noise = noise * scale

        return wav + noise

    def _add_ship_horn_noise(self, wav, samples, channels, device, snr_db=12):
        duration = samples / self.sample_rate
        min_total_duration = 0.8 * duration
        num_bursts = random.randint(2, 3)
        burst_duration = int((min_total_duration / num_bursts) * self.sample_rate)
        noise = torch.zeros(wav.shape[0], channels, samples, device=device)
        burst_starts = random.sample(range(samples - burst_duration), num_bursts)

        for start in burst_starts:
            end = min(start + burst_duration, samples)
            t = torch.linspace(0, (end - start) / self.sample_rate, end - start, device=device)
            freq = random.uniform(100, 200)
            burst_noise = (
                torch.sin(2 * np.pi * freq * t) +
                0.4 * torch.sin(2 * np.pi * freq * 2 * t)
            ) * torch.exp(-t * 2.0)
            burst_noise = burst_noise.reshape(1, 1, end - start).repeat(wav.shape[0], channels, 1)
            noise[:, :, start:end] += burst_noise.squeeze(-1)

        signal_power = torch.mean(wav ** 2)
        noise_power = torch.mean(noise ** 2)
        if noise_power > 0:
            scale = torch.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
            noise = noise * scale

        return wav + noise

    def _add_electric_buzz_noise(self, wav, samples, channels, device, snr_db=12):
        duration = samples / self.sample_rate
        min_total_duration = 0.8 * duration
        num_bursts = random.randint(2, 3)
        burst_duration = int((min_total_duration / num_bursts) * self.sample_rate)
        noise = torch.zeros(wav.shape[0], channels, samples, device=device)
        burst_starts = random.sample(range(samples - burst_duration), num_bursts)

        for start in burst_starts:
            end = min(start + burst_duration, samples)
            t = torch.linspace(0, (end - start) / self.sample_rate, end - start, device=device)
            freq = random.uniform(2000, 5000)
            burst_noise = torch.sin(2 * np.pi * freq * t + torch.randn(end - start, device=device) * 0.2)
            burst_noise = burst_noise.reshape(1, 1, end - start).repeat(wav.shape[0], channels, 1)
            noise[:, :, start:end] += burst_noise.squeeze(-1)

        signal_power = torch.mean(wav ** 2)
        noise_power = torch.mean(noise ** 2)
        if noise_power > 0:
            scale = torch.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
            noise = noise * scale

        return wav + noise

    def _add_wind_noise(self, wav, samples, channels, device, snr_db=10):
        duration = samples / self.sample_rate
        min_total_duration = 0.8 * duration
        num_bursts = random.randint(2, 3)
        burst_duration = int((min_total_duration / num_bursts) * self.sample_rate)
        noise = torch.zeros(wav.shape[0], channels, samples, device=device)
        burst_starts = random.sample(range(samples - burst_duration), num_bursts)

        for start in burst_starts:
            end = min(start + burst_duration, samples)
            t = torch.linspace(0, (end - start) / self.sample_rate, end - start, device=device)
            base_noise = torch.randn(wav.shape[0], channels, end - start, device=device)
            fft = torch.fft.rfft(base_noise, dim=-1)
            freqs = torch.linspace(0, self.sample_rate / 2, fft.shape[-1], device=device)
            mask = (freqs <= 500) * torch.exp(-freqs / 200)
            fft = fft * mask.float()
            burst_noise = torch.fft.irfft(fft, n=end - start, dim=-1)
            burst_noise *= (1 + 0.3 * torch.sin(2 * np.pi * 0.5 * t)).reshape(1, 1, -1)
            noise[:, :, start:end] += burst_noise

        signal_power = torch.mean(wav ** 2)
        noise_power = torch.mean(noise ** 2)
        if noise_power > 0:
            scale = torch.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
            noise = noise * scale

        return wav + noise

    def _add_crowd_noise(self, wav, samples, channels, device, snr_db=10):
        duration = samples / self.sample_rate
        min_total_duration = 0.8 * duration
        num_bursts = random.randint(2, 3)
        burst_duration = int((min_total_duration / num_bursts) * self.sample_rate)
        noise = torch.zeros(wav.shape[0], channels, samples, device=device)
        burst_starts = random.sample(range(samples - burst_duration), num_bursts)

        for start in burst_starts:
            end = min(start + burst_duration, samples)
            t = torch.linspace(0, (end - start) / self.sample_rate, end - start, device=device)
            low_freq = random.uniform(100, 300)
            burst_noise = torch.sin(2 * np.pi * low_freq * t) * torch.randn(end - start, device=device) * 0.5
            high_noise = torch.randn(wav.shape[0], channels, end - start, device=device)
            fft = torch.fft.rfft(high_noise, dim=-1)
            freqs = torch.linspace(0, self.sample_rate / 2, fft.shape[-1], device=device)
            mask = (freqs >= 2000) & (freqs <= 8000)
            fft = fft * mask.float()
            high_noise = torch.fft.irfft(fft, n=end - start, dim=-1)
            burst_noise = burst_noise.reshape(1, 1, -1) + 0.3 * high_noise
            noise[:, :, start:end] += burst_noise

        signal_power = torch.mean(wav ** 2)
        noise_power = torch.mean(noise ** 2)
        if noise_power > 0:
            scale = torch.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
            noise = noise * scale

        return wav + noise

    def _add_mechanical_noise(self, wav, samples, channels, device, snr_db=12):
        duration = samples / self.sample_rate
        min_total_duration = 0.8 * duration
        num_bursts = random.randint(2, 3)
        burst_duration = int((min_total_duration / num_bursts) * self.sample_rate)
        noise = torch.zeros(wav.shape[0], channels, samples, device=device)
        burst_starts = random.sample(range(samples - burst_duration), num_bursts)

        for start in burst_starts:
            end = min(start + burst_duration, samples)
            t = torch.linspace(0, (end - start) / self.sample_rate, end - start, device=device)
            knock = torch.zeros(wav.shape[0], channels, end - start, device=device)
            freq = random.uniform(2, 5)
            step = int(self.sample_rate / freq)
            indices = torch.arange(0, end - start, step, device=device)
            for idx in indices:
                knock[:, :, idx:idx + 1] = torch.randn(wav.shape[0], channels, 1, device=device) * 0.5
            resonance_freq = random.uniform(1000, 3000)
            resonance = torch.sin(2 * np.pi * resonance_freq * t) * torch.exp(-t * 5.0)
            burst_noise = knock + 0.3 * resonance.reshape(1, 1, -1)
            noise[:, :, start:end] += burst_noise

        signal_power = torch.mean(wav ** 2)
        noise_power = torch.mean(noise ** 2)
        if noise_power > 0:
            scale = torch.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
            noise = noise * scale

        return wav + noise

    def _add_water_flow_noise(self, wav, samples, channels, device, snr_db=10):
        duration = samples / self.sample_rate
        min_total_duration = 0.8 * duration
        num_bursts = random.randint(2, 3)
        burst_duration = int((min_total_duration / num_bursts) * self.sample_rate)
        noise = torch.zeros(wav.shape[0], channels, samples, device=device)
        burst_starts = random.sample(range(samples - burst_duration), num_bursts)

        for start in burst_starts:
            end = min(start + burst_duration, samples)
            burst_noise = torch.randn(wav.shape[0], channels, end - start, device=device)
            fft = torch.fft.rfft(burst_noise, dim=-1)
            freqs = torch.linspace(0, self.sample_rate / 2, fft.shape[-1], device=device)
            mask = torch.exp(-freqs / 4000)
            fft = fft * mask.float()
            burst_noise = torch.fft.irfft(fft, n=end - start, dim=-1)
            t = torch.linspace(0, (end - start) / self.sample_rate, end - start, device=device)
            burst_noise *= (1 + 0.2 * torch.sin(2 * np.pi * 0.3 * t)).reshape(1, 1, -1)
            noise[:, :, start:end] += burst_noise

        signal_power = torch.mean(wav ** 2)
        noise_power = torch.mean(noise ** 2)
        if noise_power > 0:
            scale = torch.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
            noise = noise * scale

        return wav + noise

    def _add_violin_noise(self, wav, samples, channels, device, snr_db=12):
        duration = samples / self.sample_rate
        min_total_duration = 0.8 * duration
        num_bursts = random.randint(2, 3)
        burst_durations = [random.uniform(2, 6) for _ in range(num_bursts)]
        total_burst_duration = sum(burst_durations)
        if total_burst_duration < min_total_duration:
            scale = min_total_duration / total_burst_duration
            burst_durations = [d * scale for d in burst_durations]
        burst_samples = [int(d * self.sample_rate) for d in burst_durations]
        t = torch.linspace(0, duration, samples, device=device)
        noise = torch.zeros(wav.shape[0], channels, samples, device=device)
        available_starts = list(range(samples - max(burst_samples)))
        burst_starts = []
        for burst_duration in burst_samples:
            if available_starts:
                start = random.choice(available_starts)
                burst_starts.append(start)
                start_min = max(0, start - burst_duration)
                start_max = min(samples, start + burst_duration * 2)
                available_starts = [s for s in available_starts if s < start_min or s > start_max]

        major_melodies = [
            [0, 2, 4, 5, 7, 9, 7, 5, 4, 2, 0],
            [4, 5, 7, 9, 7, 5, 4, 2, 0, 2, 4],
            [7, 9, 11, 12, 11, 9, 7, 5, 4, 5, 7],
        ]
        minor_melodies = [
            [0, 2, 3, 5, 7, 8, 7, 5, 3, 2, 0],
            [3, 5, 7, 8, 7, 5, 3, 2, 0, 2, 3],
            [7, 8, 10, 12, 10, 8, 7, 5, 3, 5, 7],
        ]
        major_chords = [
            [[0, 4, 7], [5, 9, 12], [7, 11, 14], [0, 4, 7]],
            [[0, 4, 7], [9, 12, 16], [5, 9, 12], [0, 4, 7]],
            [[7, 11, 14], [0, 4, 7], [5, 9, 12], [7, 11, 14]],
        ]
        minor_chords = [
            [[0, 3, 7], [5, 8, 12], [7, 10, 14], [0, 3, 7]],
            [[0, 3, 7], [8, 12, 15], [5, 8, 12], [0, 3, 7]],
            [[7, 10, 14], [0, 3, 7], [8, 12, 15], [7, 10, 14]],
        ]

        for start, burst_duration in zip(burst_starts, burst_samples):
            end = min(start + burst_duration, samples)
            num_notes = random.randint(8, 16)
            note_duration = int((end - start) / num_notes)
            melody_type = random.choice(['major', 'minor'])
            melody = major_melodies if melody_type == 'major' else minor_melodies
            chords = major_chords if melody_type == 'major' else minor_chords
            note_sequence = random.choice(melody)
            chord_sequence = random.choice(chords)

            chord_change_interval = random.randint(2, 4)
            num_chords = max(1, num_notes // chord_change_interval)
            chord_duration = int((end - start) / num_chords)

            for i in range(num_chords):
                chord_start = start + i * chord_duration
                chord_end = min(chord_start + chord_duration, end)
                chord = chord_sequence[i % len(chord_sequence)]
                base_freq = 440 * 2 ** (random.randint(-24, 0) / 12)

                t_segment = t[chord_start:chord_end]
                chord_wave = torch.zeros_like(t_segment)
                for note in chord:
                    freq = base_freq * 2 ** (note / 12)
                    note_wave = (
                        torch.sin(2 * np.pi * freq * t_segment) +
                        0.5 * torch.sin(2 * np.pi * (freq * 2) * t_segment) +
                        0.3 * torch.sin(2 * np.pi * (freq * 3) * t_segment)
                    )
                    chord_wave += note_wave
                envelope = torch.exp(-(t_segment - t_segment[0]) * 1.0) * (
                    1 - torch.exp(-(t_segment - t_segment[0]) * 5.0)
                )
                chord_segment = chord_wave * envelope * 0.6

                for c in range(channels):
                    noise[:, c, chord_start:chord_end] += chord_segment

            for i in range(num_notes):
                note_start = start + i * note_duration
                note_end = min(note_start + note_duration, end)
                note = note_sequence[i % len(note_sequence)]
                base_freq = 440 * 2 ** (random.randint(-12, 12) / 12)
                freq = base_freq * 2 ** (note / 12)

                t_segment = t[note_start:note_end]
                note_wave = (
                    torch.sin(2 * np.pi * freq * t_segment + torch.randn(1, device=device) * 0.05) +
                    0.4 * torch.sin(2 * np.pi * (freq * 2) * t_segment) +
                    0.2 * torch.sin(2 * np.pi * (freq * 3) * t_segment)
                )
                vibrato = torch.sin(2 * np.pi * 6 * t_segment) * 0.03
                note_wave *= torch.exp(vibrato)
                envelope = torch.exp(-(t_segment - t_segment[0]) * 2.0) * (
                    1 - torch.exp(-(t_segment - t_segment[0]) * 10.0)
                )
                dynamic = 1 + 0.2 * torch.sin(2 * np.pi * 0.1 * t_segment)
                segment = note_wave * envelope * dynamic

                for c in range(channels):
                    noise[:, c, note_start:note_end] += segment

            print(f"Violin burst start: {start}, end: {end}, melody: {note_sequence}, chords: {chord_sequence}")

        signal_power = torch.mean(wav ** 2)
        noise_power = torch.mean(noise ** 2)
        if noise_power > 0:
            scale = torch.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
            noise = noise * scale

        return wav + noise

    def _add_external_audio_noise(self, wav, samples, channels, device, snr_db=12):
        if not self.noise_files:
            print("无可用音效文件，跳过外部音效增强")
            return wav

        duration = samples / self.sample_rate
        min_total_duration = 0.8 * duration
        num_bursts = random.randint(2, 3)
        burst_duration = int((min_total_duration / num_bursts) * self.sample_rate)
        noise = torch.zeros(wav.shape[0], channels, samples, device=device)
        burst_starts = random.sample(range(samples - burst_duration), num_bursts)

        for start in burst_starts:
            end = min(start + burst_duration, samples)
            noise_file = random.choice(self.noise_files)
            try:
                noise_waveform, noise_sr = torchaudio.load(noise_file)
                if noise_sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(noise_sr, self.sample_rate)
                    noise_waveform = resampler(noise_waveform)
                if noise_waveform.shape[0] > channels:
                    noise_waveform = noise_waveform[:channels, :]
                elif noise_waveform.shape[0] < channels:
                    noise_waveform = noise_waveform.repeat(channels // noise_waveform.shape[0], 1)
                noise_length = noise_waveform.shape[-1]
                segment_length = min(end - start, noise_length)
                if noise_length > segment_length:
                    offset = random.randint(0, noise_length - segment_length)
                    noise_segment = noise_waveform[:, offset:offset + segment_length]
                else:
                    noise_segment = noise_waveform[:, :segment_length]
                if noise_segment.shape[-1] < end - start:
                    noise_segment = torch.nn.functional.pad(noise_segment, (0, end - start - noise_segment.shape[-1]))
                elif noise_segment.shape[-1] > end - start:
                    noise_segment = noise_segment[:, :end - start]
                noise_segment = noise_segment.unsqueeze(0).to(device)
                noise[:, :, start:end] += noise_segment
            except Exception as e:
                print(f"加载音效文件 {noise_file} 失败: {e}, 跳过")
                continue

        signal_power = torch.mean(wav ** 2)
        noise_power = torch.mean(noise ** 2)
        if noise_power > 0:
            scale = torch.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
            noise = noise * scale

        return wav + noise
class VocalAugmentDataset(Dataset):
    """
    自定义 Dataset，用于加载音频文件，切分为 2 秒片段，应用增强并返回增强和原始片段及噪声类型标签。
    """
    def __init__(self, audio_dir, sample_rate=44100, segment_length=2.0, augmentor=None, pad_short=True, extensions=('.wav', '.mp3', '.flac', '.m4a')):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sample_rate)
        self.augmentor = augmentor
        self.pad_short = pad_short
        self.extensions = extensions

        # 收集所有音频文件
        self.audio_files = [
            f for f in self.audio_dir.rglob('*')
            if f.is_file() and f.suffix.lower() in self.extensions
        ]
        if not self.audio_files:
            raise ValueError(f"No audio files found in {audio_dir} with extensions {extensions}")

        # 计算每个文件的片段数
        self.segments = []
        for audio_file in self.audio_files:
            try:
                info = torchaudio.info(audio_file)
                duration = info.num_frames / info.sample_rate
                num_segments = math.ceil(duration / self.segment_length) if self.pad_short else math.floor(duration / self.segment_length)
                for i in range(num_segments):
                    self.segments.append((audio_file, i))
            except Exception as e:
                print(f"Warning: Failed to process {audio_file}: {e}")

        if not self.segments:
            raise ValueError("No valid audio segments found")

    def __len__(self):
        """返回数据集中的总片段数"""
        return len(self.segments)

    def __getitem__(self, idx):
        """
        获取第 idx 个片段，应用增强并返回增强和原始片段及噪声类型标签。

        返回：
            tuple: (augmented_segment, original_segment, applied_augs)
                - augmented_segment (torch.Tensor): 增强后的音频片段，形状 [2, segment_samples]
                - original_segment (torch.Tensor): 原始音频片段，形状 [2, segment_samples]
                - applied_augs (torch.Tensor): 噪声类型标签，形状 [num_noise_types]
        """
        audio_file, segment_idx = self.segments[idx]

        # 加载音频
        try:
            wav, sr = torchaudio.load(audio_file)
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            # 返回双通道零张量及零标签
            return torch.zeros(2, self.segment_samples), torch.zeros(2, self.segment_samples), torch.zeros(len(self.augmentor.noise_types))

        # 重采样到目标采样率
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = resampler(wav)

        # 转换为双通道
        num_channels = wav.shape[0]
        if num_channels == 1:
            wav = wav.repeat(2, 1)  # [2, samples]
        elif num_channels > 2:
            wav = wav[:2, :]  # [2, samples]

        # 计算片段的起始和结束样本
        start_sample = segment_idx * self.segment_samples
        end_sample = start_sample + self.segment_samples

        # 提取片段
        if end_sample <= wav.shape[-1]:
            segment = wav[:, start_sample:end_sample]
        else:
            segment = wav[:, start_sample:]
            if self.pad_short:
                pad_length = self.segment_samples - segment.shape[-1]
                segment = torch.nn.functional.pad(segment, (0, pad_length))
            else:
                return torch.zeros(2, self.segment_samples), torch.zeros(2, self.segment_samples), torch.zeros(len(self.augmentor.noise_types))

        # 保存原始片段
        original_segment = segment.clone()

        # 应用增强
        if self.augmentor is not None:
            segment = segment.unsqueeze(0)  # [1, 2, samples]
            augmented_segment, applied_augs = self.augmentor(segment)
            augmented_segment = augmented_segment.squeeze(0)  # [2, samples]
        else:
            augmented_segment = segment.clone()
            applied_augs = torch.zeros(len(self.augmentor.noise_types))

        return augmented_segment, original_segment, applied_augs

    def get_audio_files(self):
        """返回所有音频文件的路径列表"""
        return [str(f) for f in self.audio_files]

def train_unet(
    audio_dir,
    save_dir,
    sample_rate=44100,
    segment_length=2.0,
    batch_size=4,
    num_epochs=50,
    lr=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    n_fft=2048,
    hop_length=512,
    use_complex=False,
    validate_every=5,
    save_every=5
):
    """
    训练 UNetAudio 模型以分离人声并识别噪声类型。

    参数：
        audio_dir (str): 音频文件所在文件夹路径
        save_dir (str): 保存模型权重和日志的文件夹
        sample_rate (int): 采样率
        segment_length (float): 片段长度（秒）
        batch_size (int): 批次大小
        num_epochs (int): 训练轮数
        lr (float): 学习率
        device (str): 训练设备（'cuda' 或 'cpu'）
        n_fft (int): STFT 的 FFT 点数
        hop_length (int): STFT 的跳跃长度
        use_complex (bool): 是否使用复数频谱图
        validate_every (int): 每多少个 epoch 进行一次验证
        save_every (int): 每多少个 epoch 保存一次模型
    """
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 初始化增强器和数据集
    augmentor = VocalAugmentor(sample_rate=sample_rate, p=0.7, max_augmentations=12)
    dataset = VocalAugmentDataset(
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        augmentor=augmentor,
        pad_short=True,
        extensions=('.wav', '.mp3', '.flac')
    )

    # 划分训练和验证集
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 初始化模型
    in_channels = 2
    out_channels = 2
    model = UNetAudio(in_channels=in_channels, out_channels=out_channels, num_noise_types=len(augmentor.noise_types)).to(device)
    model.load_state_dict(torch.load(r"E:\美赛\2025年第十五届MathorCup数学应用挑战赛赛题\C题\录音\model_c.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    noise_criterion = nn.BCELoss()  # 用于噪声分类

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_noise_loss = 0.0
        train_steps = 0

        for augmented_batch, original_batch, applied_augs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            augmented_batch = augmented_batch.to(device)
            original_batch = original_batch.to(device)
            applied_augs = applied_augs.to(device)

            # 转换为频谱图
            aug_spec, aug_phase = wav_to_spectrogram(augmented_batch, n_fft, hop_length, use_complex)
            orig_spec, _ = wav_to_spectrogram(original_batch, n_fft, hop_length, use_complex)

            # 前向传播
            optimizer.zero_grad()
            mask, noise_probs = model(aug_spec)
            pred_spec = aug_spec * mask

            # 计算频谱图损失
            loss = criterion(pred_spec, orig_spec)

            # 计算时间域损失
            pred_wav = spectrogram_to_wav(pred_spec, aug_phase, n_fft, hop_length, use_complex)
            target_length = original_batch.shape[-1]
            if pred_wav.shape[-1] > target_length:
                pred_wav = pred_wav[..., :target_length]
            elif pred_wav.shape[-1] < target_length:
                pred_wav = torch.nn.functional.pad(pred_wav, (0, target_length - pred_wav.shape[-1]))

            time_loss = criterion(pred_wav, original_batch)

            # 计算噪声分类损失
            noise_loss = noise_criterion(noise_probs, applied_augs)

            # 总损失
            total_loss = loss + 0.5 * time_loss + 0.3 * noise_loss

            # 反向传播
            total_loss.backward()
            optimizer.step()

            train_loss += (loss.item() + 0.5 * time_loss.item())
            train_noise_loss += noise_loss.item()
            train_steps += 1

        avg_train_loss = train_loss / train_steps
        avg_train_noise_loss = train_noise_loss / train_steps
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Train Noise Loss: {avg_train_noise_loss:.6f}")

        # 验证
        if (epoch + 1) % validate_every == 0:
            model.eval()
            val_loss = 0.0
            val_noise_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for augmented_batch, original_batch, applied_augs in tqdm(val_loader, desc="Validating"):
                    augmented_batch = augmented_batch.to(device)
                    original_batch = original_batch.to(device)
                    applied_augs = applied_augs.to(device)

                    aug_spec, aug_phase = wav_to_spectrogram(augmented_batch, n_fft, hop_length, use_complex)
                    orig_spec, _ = wav_to_spectrogram(original_batch, n_fft, hop_length, use_complex)

                    mask, noise_probs = model(aug_spec)
                    pred_spec = aug_spec * mask
                    loss = criterion(pred_spec, orig_spec)

                    pred_wav = spectrogram_to_wav(pred_spec, aug_phase, n_fft, hop_length, use_complex)
                    target_length = original_batch.shape[-1]
                    if pred_wav.shape[-1] > target_length:
                        pred_wav = pred_wav[..., :target_length]
                    elif pred_wav.shape[-1] < target_length:
                        pred_wav = torch.nn.functional.pad(pred_wav, (0, target_length - pred_wav.shape[-1]))
                    time_loss = criterion(pred_wav, original_batch)

                    noise_loss = noise_criterion(noise_probs, applied_augs)
                    total_loss = loss + 0.5 * time_loss + 0.3 * noise_loss

                    val_loss += (loss.item() + 0.5 * time_loss.item())
                    val_noise_loss += noise_loss.item()
                    val_steps += 1

                    # 打印第一个批次的噪声分类结果
                    if val_steps == 1:
                        print("Sample Noise Classification Probabilities:")
                        for i, noise_type in enumerate(augmentor.noise_types):
                            print(f"{noise_type}: Pred={noise_probs[0, i]:.4f}, True={applied_augs[0, i]:.4f}")

            avg_val_loss = val_loss / val_steps
            avg_val_noise_loss = val_noise_loss / val_steps
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.6f}, Validation Noise Loss: {avg_val_noise_loss:.6f}")

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_dir / 'best_model_c.pth')
                print(f"Saved best model with Validation Loss: {best_val_loss:.6f}")

        # 定期保存模型
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), save_dir / f'model_c.pth')
            print(f"Saved model at epoch {epoch+1}")

    print("Training completed!")

if __name__ == "__main__":
    # 设置参数
    audio_dir = r"E:\美赛\2025年第十五届MathorCup数学应用挑战赛赛题\C题\train"
    save_dir = r"E:\美赛\2025年第十五届MathorCup数学应用挑战赛赛题\C题\录音"
    sample_rate = 44100
    segment_length = 2.0
    batch_size = 2
    num_epochs = 500
    lr = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_fft = 2048
    hop_length = 512
    use_complex = False  # 使用幅度谱图

    # 开始训练
    train_unet(
        audio_dir=audio_dir,
        save_dir=save_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        n_fft=n_fft,
        hop_length=hop_length,
        use_complex=use_complex
    )