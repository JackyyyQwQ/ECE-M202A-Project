# Audio data

Source: [DCASE2019_task4_synthetic_data](https://zenodo.org/records/2583796),
DOI `10.5281/zenodo.2583796`. The source describes 10-second mixtures generated
with Scaper from foreground events and domestic backgrounds.

The team's preparation summary reports 2,045 recordings, ten classes, resampling
from 44.1 kHz to 16 kHz mono PCM16, and a recording-level 80/10/10 split with seed
2026. These counts/formats and the actual split manifests have not yet been
checked against downloaded files in this checkout. Preserve the team's existing
split and class ordering rather than silently generating a replacement.

## Files expected by the existing loaders

Run commands from the repository root. The window loader uses these relative
paths; the sequence loader accepts a root containing the same layout:

```text
processed_audio/
  wav/*.wav
  recording_splits.csv     # filename,split; train / val / test
  classes.txt             # one class per line, fixed order
Synthetic_dataset/
  synthetic_dataset.csv   # tab-separated filename,onset,offset,event_label
```

Audio and manifests are not included. `dataset_prep.py` is described by the team
but has not been provided. The sequence loader currently accepts train/val only.

## Loaders

- `audio_dataset.py`: individual 8,000-sample windows and ten-label tensors.
- `sequence_dataset.py`: chronological windows from a selected recording.
- Hop: 800 samples. Targets describe the last 50 ms of each window. A ten-second
  recording yields 191 windows; the first 450 ms is unscored.

After placing the required local files, inspect a batch with:

```bash
python -m data.audio_dataset
```

This is a data inspection command, not training or a reproduction of reported
metrics. The original ad-hoc inspection expects a sufficiently long first recording.
For old scripts, update imports to `data.audio_dataset` / `data.sequence_dataset`.

Archive: `Synthetic_dataset.zip`, 1,190,326,055 bytes.
MD5: `06ebea7cc467ace800752bd21ec7178d`.
Use the source record for citation and license details. Keep audio out of Git.
