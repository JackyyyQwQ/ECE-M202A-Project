"""Load the author's DCASE 2019 CNN+BiGRU checkpoint with unmodified upstream code.

Upstream: https://github.com/turpaultn/DCASE2019_task4 at UPSTREAM_COMMIT, cloned
to cache/DCASE2019_task4 (ignored). Weights: checkpoints/dcase2019/dcase2019_t4_mdl
(ignored), from the author's reply in upstream issue #17.
"""

import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "cache/DCASE2019_task4"
UPSTREAM_COMMIT = "90d35b6191a5d8de05bca816ebc1a836e4a83a00"
CHECKPOINT = ROOT / "checkpoints/dcase2019/dcase2019_t4_mdl"
CHECKPOINT_SOURCE = "https://github.com/turpaultn/DCASE2019_task4/issues/17#issuecomment-1252540486"
SAMPLE_RATE = 44100
STEP_S = 8 * 511 / SAMPLE_RATE  # hop 511 samples x temporal pooling 8 = ~92.70 ms


def load_detector():
    """Return (model, scaler) with the checkpoint's training mean/std scaler."""
    revision = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f"Check out {UPSTREAM} at {UPSTREAM_COMMIT}")
    sys.path.insert(0, str(UPSTREAM / "baseline"))
    from models.CRNN import CRNN

    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = CRNN(**state["model"]["kwargs"])
    model.load(parameters=state["model"]["state_dict"])
    model.eval()
    mean = np.array(state["scaler"]["mean_"])
    std = np.sqrt(np.array(state["scaler"]["mean_of_square_"]) - mean**2)
    if not (np.isfinite(std).all() and (std > 0).all()):
        raise ValueError("Invalid scaler statistics in checkpoint")
    return model, (mean, std)


def normalized_mel(wave, scaler):
    """44.1 kHz mono waveform [1, samples] -> normalized log-mel [1, 1, frames, 64]."""
    spec = librosa.stft(wave[0].numpy(), n_fft=2048, hop_length=511,
                        window=np.hamming(2048), center=True, pad_mode="reflect")
    mel = librosa.feature.melspectrogram(S=np.abs(spec), sr=SAMPLE_RATE, n_mels=64,
                                         fmin=0, fmax=22050, htk=False, norm=None)
    feature = librosa.amplitude_to_db(mel).T.astype(np.float32)
    # Upstream pads log features to 864 frames before normalization. Keep the
    # complete 11 s clip by padding it to the next multiple of the pooling factor 8.
    frames = max(864, int(np.ceil(len(feature) / 8)) * 8)
    feature = np.pad(feature, ((0, frames - len(feature)), (0, 0)))
    mean, std = scaler
    return torch.from_numpy(((feature - mean) / std).astype(np.float32))[None, None]
