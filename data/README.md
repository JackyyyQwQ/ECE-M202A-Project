# Audio data

Source: [DCASE2019_task4_synthetic_data](https://zenodo.org/records/2583796),
DOI `10.5281/zenodo.2583796`. The source describes 10-second mixtures generated
with Scaper from foreground events and domestic backgrounds.

The team's preparation summary reports 2,045 recordings, ten classes, resampling
from 44.1 kHz to 16 kHz mono PCM16, and a recording-level 80/10/10 split with seed
2026. The original archive count and 44.1 kHz mono format are now verified.
The team's processed 16 kHz files and original split manifests have not been
recovered. Preserve the team's split and class ordering rather than silently
generating a replacement.

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


## DCASE2019 synthetic corpus

The 2,045 synthetic recordings (44.1 kHz mono, 6,032 events) were downloaded
and checked (archive MD5, member CRC32), then deleted locally. They are planned
as training audio for the compact Nano encoder; download them again with
`python scripts/fetch_dcase_pilot.py --all`. Earlier 2023-model scores on this
corpus used the old feature scaling and were removed. No original custom-model
train/validation/test split was recovered.

## Real recordings for the official detectors

- **Public evaluation (692 recordings, final test set):**
  `python scripts/fetch_public_eval.py` downloads
  [Zenodo 4560759](https://zenodo.org/records/4560759) into
  `data/raw/dcase2019_public_eval/` and writes a SHA-256 manifest. Use it only to
  score frozen configurations; do not tune thresholds, lookahead or the controller on it.
- **Validation (1,168 recordings, development set):** `python scripts/fetch_validation.py`
  cuts each clip from YouTube at the official start/end times (official DESED
  procedure; needs `requirements-download.txt` and ffmpeg) into
  `data/raw/dcase2019_validation_youtube/`. It is resumable; removed or private
  videos are listed in `missing.tsv` (an earlier redistribution reports ~83%
  coverage). Re-downloaded audio may differ from the organizers' copy; report the
  available count with any result. (An unofficial mirror's clips had inconsistent
  durations, so they were discarded.)
