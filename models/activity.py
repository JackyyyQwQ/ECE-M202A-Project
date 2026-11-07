"""Small window CNN + causal recording-level GRU, with no pretrained weights."""

import torch
from torch import nn
from torch.nn import functional as F


class LogSpectrogram(nn.Module):
    """Fixed per-window frontend: no future audio and no centered STFT padding."""

    def __init__(self):
        super().__init__()
        self.register_buffer("hann", torch.hann_window(400))

    def forward(self, windows):
        spectrum = torch.stft(
            windows, n_fft=400, hop_length=200, win_length=400,
            window=self.hann, center=False, return_complex=True,
        )
        log_power = spectrum.abs().square().clamp_min(1e-10).log()
        return F.adaptive_avg_pool2d(log_power[:, None], (32, 16))


class WindowEncoder(nn.Module):
    """Model 1: waveform -> fixed spectrum -> CNN -> learned feature vector."""

    def __init__(self, feature_dim=32):
        super().__init__()
        self.frontend = LogSpectrogram()
        self.register_buffer("mean", torch.zeros(1, 1, 32, 1))
        self.register_buffer("std", torch.ones(1, 1, 32, 1))
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2)), nn.Flatten(), nn.Linear(128, feature_dim),
        )

    def encode_spectrograms(self, spectra):
        return self.cnn((spectra - self.mean) / self.std)

    def forward(self, windows):
        return self.encode_spectrograms(self.frontend(windows))


class CausalActivityModel(nn.Module):
    """Model 2 is a unidirectional GRU(32 hidden units) plus ten logits.

    Each call starts with zero hidden state for each recording. Every step
    depends only on that recording's current and earlier window features.
    """

    def __init__(self, feature_dim=32, n_classes=10):
        super().__init__()
        self.encoder = WindowEncoder(feature_dim)
        self.gru = nn.GRU(feature_dim, 32, batch_first=True)
        self.classifier = nn.Linear(32, n_classes)

    def _predict(self, features, batch_size, steps):
        sequence, _ = self.gru(features.reshape(batch_size, steps, -1))
        return self.classifier(sequence)

    def forward(self, windows):
        batch_size, steps, samples = windows.shape
        return self._predict(self.encoder(windows.reshape(-1, samples)), batch_size, steps)

    def forward_spectrograms(self, spectra):
        """Same learnable computation; cache only the fixed frontend inputs."""
        batch_size, steps = spectra.shape[:2]
        features = self.encoder.encode_spectrograms(spectra.flatten(0, 1))
        return self._predict(features, batch_size, steps)


def model_size(model):
    """Storage estimates exclude optimizer state and runtime activations."""
    def size(module):
        parameters = sum(p.numel() for p in module.parameters())
        weight_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in module.buffers())
        return {"parameters": parameters, "fp32_weight_bytes": weight_bytes,
                "fp32_weight_KiB": weight_bytes / 1024, "buffer_bytes": buffer_bytes}
    return {"model_1": size(model.encoder), "whole_model": size(model)}
