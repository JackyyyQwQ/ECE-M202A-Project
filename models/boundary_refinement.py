"""An optional boundary-refinement head; the initial CNN/GRU stays frozen."""

from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RequestSpectrogram(nn.Module):
    """Fixed frontend for exactly 3,200 samples, no padding or centering."""

    def __init__(self):
        super().__init__()
        self.register_buffer("hann", torch.hann_window(400))

    def forward(self, audio):
        if audio.shape[-1] != 3200:
            raise ValueError("Refinement requires exactly 200 ms at 16 kHz")
        z = torch.stft(audio, n_fft=400, hop_length=80, window=self.hann,
                       center=False, return_complex=True)
        return F.adaptive_avg_pool2d(z.abs().square().clamp_min(1e-10).log()[:, None], (32, 32))


class BoundaryRefiner(nn.Module):
    """Fuse requested audio with 32 CNN features, 32 GRU states and identity.

    Context: [CNN(32), GRU(32), class score(1), class one-hot(10), kind one-hot(2)].
    Outputs: support logit and correction in [-1,1], representing +/-100 ms.
    """

    def __init__(self, use_audio=True):
        super().__init__()
        self.use_audio = use_audio
        self.register_buffer("context_mean", torch.zeros(65))
        self.register_buffer("context_std", torch.ones(65))
        self.register_buffer("audio_mean", torch.zeros(1, 1, 32, 1))
        self.register_buffer("audio_std", torch.ones(1, 1, 32, 1))
        if use_audio:
            self.audio_encoder = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(), nn.Linear(512, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(77 + (32 if use_audio else 0), 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, spectra, context):
        normalized = torch.cat([(context[:, :65] - self.context_mean) / self.context_std, context[:, 65:]], dim=1)
        if self.use_audio:
            encoded = self.audio_encoder((spectra - self.audio_mean) / self.audio_std)
            normalized = torch.cat([normalized, encoded], dim=1)
        output = self.fusion(normalized)
        return output[:, 0], output[:, 1].tanh()


def decision_step(available_time, frame_ends):
    """Return the exact decision frame; never take later feature context."""
    if available_time is None:
        raise ValueError("Unconfirmed terminal boundary has no decision frame")
    index = int(np.argmin(np.abs(frame_ends - available_time)))
    if abs(float(frame_ends[index]) - available_time) > 1e-10:
        raise ValueError("Decision time does not align with a feature frame")
    return index


def apply_corrections(events, requests, support, corrections, threshold, coverage):
    """Prediction-only policy; annotations are not an argument.

    Skipped/low-support/invalid corrections retain the initial estimate. If
    corrections invert an event, both endpoints revert. Censor flags are fixed.
    """
    updates, decisions = {}, []
    for request in requests:
        index = request["cache_index"]
        initial = request["predicted_time"]
        row = {"request_id": request["request_id"], "prediction_id": request["prediction_id"],
               "filename": request["filename"], "label": request["label"], "boundary": request["boundary"],
               "initial_time": initial, "refined_time": initial, "support": None,
               "proposed_correction_ms": None, "applied_correction_ms": 0.0, "status": request["skip_reason"]}
        if index >= 0:
            row["support"] = float(support[index])
            delta = float(corrections[index]) * .1
            row["proposed_correction_ms"] = delta * 1000
            proposed = initial + delta
            limits = coverage[request["filename"]]
            if support[index] < threshold:
                row["status"] = "abstain_low_support"
            elif not (request["requested_start"] <= proposed <= request["requested_end"]
                      and limits["scored_start"] < proposed < limits["scored_end"]):
                row["status"] = "abstain_outside_bounds"
            else:
                row.update(refined_time=proposed, applied_correction_ms=delta * 1000, status="accepted")
        updates[(row["prediction_id"], row["boundary"])] = row
        decisions.append(row)
    refined = []
    for event in events:
        onset, offset = updates[(event.event_id, "onset")], updates[(event.event_id, "offset")]
        if onset["refined_time"] >= offset["refined_time"]:
            for row in (onset, offset):
                if row["status"] == "accepted":
                    row.update(refined_time=row["initial_time"], applied_correction_ms=0.0, status="abstain_invalid_interval")
        refined.append(replace(event, onset=onset["refined_time"], offset=offset["refined_time"],
                               visible_onset=onset["refined_time"], visible_offset=offset["refined_time"]))
    return refined, decisions


def support_metrics(probabilities, labels, known, threshold):
    predicted = probabilities[known] >= threshold
    actual = labels[known] > .5
    tp, fp, fn = int((predicted & actual).sum()), int((predicted & ~actual).sum()), int((~predicted & actual).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "known": int(known.sum()), "positive": int(actual.sum()),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def choose_gate(probabilities, labels, known):
    """Development-only support F1; ties prefer precision, then higher gate."""
    rows = []
    for threshold in [round(i / 20, 2) for i in range(1, 20)] + [1.01]:
        rows.append({"threshold": threshold, **support_metrics(probabilities, labels, known, threshold)})
    best = max(rows, key=lambda r: (r["f1"], r["precision"], r["threshold"]))
    return best["threshold"], rows
