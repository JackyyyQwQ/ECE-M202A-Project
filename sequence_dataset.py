"""Whole-recording sequences using the existing AudioWindowDataset conventions."""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


SAMPLE_RATE = 16000
WINDOW_SAMPLES = 8000
HOP_SAMPLES = 800


def activity_targets(n_windows, events, n_classes):
    """Label [end - 50 ms, end) by strictly positive event overlap."""
    ends = (np.arange(n_windows) * HOP_SAMPLES + WINDOW_SAMPLES) / SAMPLE_RATE
    starts = (np.arange(n_windows) * HOP_SAMPLES + WINDOW_SAMPLES - HOP_SAMPLES) / SAMPLE_RATE
    targets = np.zeros((n_windows, n_classes), dtype=np.float32)
    for onset, offset, class_index in events:
        targets[(onset < ends) & (offset > starts), class_index] = 1
    return torch.from_numpy(targets), starts, ends


class RecordingSequenceDataset(Dataset):
    """One item is one recording; windows never cross recording boundaries.

    Only explicitly selected train/val files are opened. No audio is padded.
    """

    def __init__(self, root, split, filenames):
        self.root = Path(root)
        if split not in {"train", "val"}:
            raise ValueError("This feasibility dataset accepts only train/val")
        self.split = split
        self.filenames = list(filenames)
        self.classes = (self.root / "processed_audio/classes.txt").read_text().splitlines()
        if len(self.classes) != 10 or len(set(self.classes)) != 10:
            raise ValueError("Expected ten unique classes in their existing order")
        with (self.root / "processed_audio/recording_splits.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        if len({r["filename"] for r in rows}) != len(rows):
            raise ValueError("Duplicate recording in split manifest")
        allowed = {r["filename"] for r in rows if r["split"] == split}
        if not self.filenames or len(set(self.filenames)) != len(self.filenames):
            raise ValueError("Selected filenames must be nonempty and unique")
        if not set(self.filenames) <= allowed:
            raise ValueError("Selected recordings do not belong to the requested split")
        class_index = {name: i for i, name in enumerate(self.classes)}
        self.events = {name: [] for name in self.filenames}
        with (self.root / "Synthetic_dataset/synthetic_dataset.csv").open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["filename"] in self.events:
                    onset, offset = float(row["onset"]), float(row["offset"])
                    if not 0 <= onset < offset:
                        raise ValueError(f"Invalid event: {row}")
                    self.events[row["filename"]].append(
                        (onset, offset, class_index[row["event_label"]])
                    )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        audio, sr = sf.read(self.root / "processed_audio/wav" / filename, dtype="float32")
        if sr != SAMPLE_RATE or audio.ndim != 1:
            raise ValueError(f"Expected 16 kHz mono: {filename}")
        if len(audio) < WINDOW_SAMPLES or not np.isfinite(audio).all():
            raise ValueError(f"Too short or nonfinite audio: {filename}")
        windows = torch.from_numpy(audio).unfold(0, WINDOW_SAMPLES, HOP_SAMPLES)
        targets, starts, ends = activity_targets(len(windows), self.events[filename], len(self.classes))
        return {
            "filename": filename,
            "windows": windows,
            "targets": targets,
            "frame_starts": starts,
            "frame_ends": ends,
            "duration": len(audio) / sr,
        }
