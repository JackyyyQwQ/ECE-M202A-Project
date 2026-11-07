
from pathlib import Path
from bisect import bisect_right

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader


class AudioWindowDataset(Dataset):

    def __init__(self, split="train"):
        # Configuration
        self.sr = 16000
        self.window_size = 8000  # 500 ms
        self.hop_size = 800      # 50 ms

        root = Path("processed_audio")
        self.wav_dir = root / "wav"

        # Load class names
        self.classes = (root / "classes.txt").read_text().splitlines()
        class_to_idx = {
            name: i for i, name in enumerate(self.classes)
        }

        # Select recordings belonging to this split
        manifest = pd.read_csv(root / "recording_splits.csv")
        self.filenames = manifest.loc[
            manifest["split"] == split, "filename"
        ].tolist()

        # Load original event annotations
        annotations = pd.read_csv(
            "Synthetic_dataset/synthetic_dataset.csv",
            sep="\t"
        )

        # Group events by recording
        self.events = {}

        for row in annotations.itertuples(index=False):
            self.events.setdefault(row.filename, []).append(
                (row.onset, row.offset, class_to_idx[row.event_label])
            )

        # Build an index of all available windows
        self.cumulative = [0]

        for filename in self.filenames:
            info = sf.info(self.wav_dir / filename)

            assert info.samplerate == self.sr
            assert info.channels == 1

            n_windows = max(
                0,
                (info.frames - self.window_size) // self.hop_size + 1
            )

            self.cumulative.append(
                self.cumulative[-1] + n_windows
            )

    def __len__(self):
        return self.cumulative[-1]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # Find which recording contains this window
        file_idx = bisect_right(self.cumulative, idx) - 1
        window_idx = idx - self.cumulative[file_idx]

        filename = self.filenames[file_idx]

        start = window_idx * self.hop_size
        end = start + self.window_size

        # Read only the required 500 ms
        with sf.SoundFile(self.wav_dir / filename) as f:
            f.seek(start)
            audio = f.read(
                self.window_size,
                dtype="float32"
            )

        # Latest 50 ms prediction interval
        frame_start = (end - self.hop_size) / self.sr
        frame_end = end / self.sr

        # Generate the 10-class label
        label = np.zeros(len(self.classes), dtype=np.float32)

        for onset, offset, class_idx in self.events.get(filename, []):
            if onset < frame_end and offset > frame_start:
                label[class_idx] = 1.0

        return torch.from_numpy(audio), torch.from_numpy(label)


if __name__ == "__main__":
    # Test the Dataset
    dataset = AudioWindowDataset(split="train")

    print("Total training windows:", len(dataset))

    x, y = dataset[0]

    print("Single audio input:", x.shape)
    print("Single label:", y.shape)
    print("Label values:", y)

    # Test the DataLoader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    x_batch, y_batch = next(iter(loader))

    print("Batch audio shape:", x_batch.shape)
    print("Batch label shape:", y_batch.shape)

    for i in range(min(1000, len(dataset))):
        x, y = dataset[i]

        if y.sum() > 0:
            print("First positive example:", i)
            print("Labels:", y)
            break
    else:
        print("No positive labels in the first 1000 windows")

    i = 85

    filename = dataset.filenames[0]
    x, y = dataset[i]

    print("Recording:", filename)
    print("Audio window:", 85 * 0.05, "to", 85 * 0.05 + 0.5)
    print("Active classes:",
        [dataset.classes[j] for j in range(len(y)) if y[j] == 1])

    print("Original annotations:")
    for onset, offset, class_idx in dataset.events[filename]:
        print(dataset.classes[class_idx], onset, offset)