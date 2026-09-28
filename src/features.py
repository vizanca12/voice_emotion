"""Audio front-end: waveform -> log-mel spectrogram.

This module is imported by both the training scripts and the inference server.
That is deliberate: if the app computed features even slightly differently from
training, the model would be fed a distribution it never saw and would quietly
lose accuracy with no error anywhere. One implementation, one source of truth.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

from . import config as C


def load_wav(path) -> torch.Tensor:
    """Read a file, downmix to mono, resample to SAMPLE_RATE. Returns 1-D float32."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != C.SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, C.SAMPLE_RATE)
    return wav.squeeze(0).float()


def fix_length(wav: torch.Tensor, n_samples: int = C.N_SAMPLES) -> torch.Tensor:
    """Crop to the most energetic n_samples window, or zero-pad centred.

    RAVDESS clips carry roughly half a second of silence at each end, and the
    spoken part sits somewhere in the middle. Picking the loudest window trims
    the silence and centres the speech in one pass, with no threshold to tune.
    """
    n = wav.numel()
    if n == n_samples:
        return wav
    if n < n_samples:
        pad = n_samples - n
        left = pad // 2
        return torch.nn.functional.pad(wav, (left, pad - left))

    # Sliding-window energy via a cumulative sum: O(n) instead of O(n * window).
    energy = torch.cumsum(wav.pow(2), dim=0)
    energy = torch.cat([torch.zeros(1, dtype=energy.dtype), energy])
    window_energy = energy[n_samples:] - energy[:-n_samples]
    start = int(torch.argmax(window_energy).item())
    return wav[start:start + n_samples]


class LogMel(nn.Module):
    """Log-mel spectrogram, per-example standardised. Output: (1, N_MELS, N_FRAMES)."""

    def __init__(self) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=C.SAMPLE_RATE,
            n_fft=C.N_FFT,
            win_length=C.WIN_LENGTH,
            hop_length=C.HOP_LENGTH,
            f_min=C.F_MIN,
            f_max=C.F_MAX,
            n_mels=C.N_MELS,
            power=2.0,
            center=True,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=C.TOP_DB)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)          # (1, n_samples)
        spec = self.to_db(self.mel(wav))    # (B, n_mels, n_frames)
        # Per-example standardisation: removes recording gain and microphone
        # colouration, which differ between the corpus and a laptop mic.
        mean = spec.mean(dim=(-2, -1), keepdim=True)
        std = spec.std(dim=(-2, -1), keepdim=True)
        spec = (spec - mean) / (std + 1e-5)
        return spec.unsqueeze(-3)           # (B, 1, n_mels, n_frames)


def spec_augment(
    spec: torch.Tensor,
    n_freq_masks: int = 2,
    freq_width: int = 8,
    n_time_masks: int = 2,
    time_width: int = 30,
    max_shift: int = 40,
) -> torch.Tensor:
    """SpecAugment + random time shift (slide 17). Training only, in place on a clone."""
    spec = spec.clone()
    n_mels, n_frames = spec.shape[-2], spec.shape[-1]

    if max_shift > 0:
        spec = torch.roll(spec, shifts=int(torch.randint(-max_shift, max_shift + 1, (1,))), dims=-1)

    for _ in range(n_freq_masks):
        w = int(torch.randint(0, freq_width + 1, (1,)))
        if w:
            f0 = int(torch.randint(0, n_mels - w, (1,)))
            spec[..., f0:f0 + w, :] = 0.0

    for _ in range(n_time_masks):
        w = int(torch.randint(0, time_width + 1, (1,)))
        if w:
            t0 = int(torch.randint(0, n_frames - w, (1,)))
            spec[..., :, t0:t0 + w] = 0.0

    return spec
