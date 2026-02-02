import math
from pathlib import Path

import torch
import torchaudio

SAMPLE_RATE = 16000


def load_audio(path: Path) -> torch.Tensor:
    wav, rate = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if rate != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, rate, SAMPLE_RATE)
    return wav.squeeze(0)


def pad_or_trim(wave: torch.Tensor, target_len: int) -> torch.Tensor:
    if wave.numel() >= target_len:
        return wave[:target_len]
    pad = target_len - wave.numel()
    return torch.nn.functional.pad(wave, (0, pad))


def make_melspec(sample_rate: int = SAMPLE_RATE, n_mels: int = 40):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=n_mels,
        center=True,
        power=2.0,
    )


def log_mel(melspec: torch.Tensor) -> torch.Tensor:
    return torch.log(melspec + 1e-6)


def energy(wave: torch.Tensor) -> float:
    return math.sqrt(torch.mean(wave.float() ** 2).item()) if wave.numel() else 0.0
