# Official DCASE2019 public evaluation: frozen 2023 CNN+BiGRU

Measured November 7, 2026. This is the **692-recording real public evaluation set**,
not official validation and not the 2,045-recording synthetic training set.
The model is still the **DCASE 2023 plain pretrained CNN+BiGRU student**. No 2019
model weights were used. No training, threshold tuning or model selection was done
on this set.

## Provenance and coverage

- Official source: [DESED_public_eval, Zenodo 4560759](https://zenodo.org/records/4560759).
- Archive: `DESED_public_eval.tar.gz`, 889,698,336 bytes.
- Official MD5 verified: `c1bf328257523f7e93244f3c1b160eb4`.
- All **692 WAVs** were extracted and SHA256 hashed. No quality-based exclusions.
- **2,765 event annotations**. Archive `public.tsv` matches the official GitHub
  [annotation table](https://github.com/turpaultn/DCASE2019_task4/blob/public/dataset/metadata/eval/public.tsv)
  row for row.
- All sources are 44.1 kHz. Channel counts: 71 mono, 616 stereo, five six-channel.
  Channels are averaged to mono and audio is resampled to 16 kHz for the model.
  Thus this is a mono-input SED experiment, not spatial localization.
- Clip durations range from four to eleven seconds. Short clips are zero-padded
  to the trained ten-second input window but padding is excluded from scoring.
  The single eleven-second recording is retained in full (172 output frames),
  including its annotation through 11 seconds. No end annotation was discarded.
- Total source duration: 6857.323 seconds. Scored
  frame intervals: **130,917**.

## Fixed scoring contract

Threshold is **>=0.5**. Primary results use raw probabilities. A separate fixed
seven-frame symmetric median filter is the secondary condition, not a tuned one.

Frame micro scores use the existing comparison-oriented 50 ms target grid,
starting at [0.45,0.50) seconds and continuing to each clip's usable duration.
Ground truth is positive on any annotation overlap. Each target takes the model's
native 64 ms bin containing its midpoint. First 450 ms are unscored for frame F1.

Standard **event macro F1** instead uses native 64 ms predictions across the full
available recording, including its beginning. `sed_eval` 0.2.1 uses optimal matching,
200 ms onset tolerance and offset tolerance max(200 ms, 20% reference duration).
A one-second segment score is also saved. These are different metrics from the
previous permissive positive-overlap event F1; their values must not be conflated.
We do not claim reproduction of any published official baseline score.

## Results

| Population / output | Frame precision | Frame recall | Frame micro F1 | Standard event macro F1 |
|---|---:|---:|---:|---:|
| All 692 / raw | 70.67% | 27.13% | 39.20% | 16.55% |
| All 692 / median-7 | 70.38% | 26.63% | 38.64% | 17.16% |
| Known-source-overlap excluded: 688 / raw | 70.57% | 27.15% | 39.21% | 16.54% |
| Known-source-overlap excluded: 688 / median-7 | 70.28% | 26.64% | 38.64% | 17.15% |

All-692 raw frame counts: **TP 31,753, FP 13,181,
FN 85,307**. Raw frame macro F1 is 32.06%;
raw one-second segment macro F1 is
38.84%.

The main finding remains **low recall**. The default detector misses many positive
frames on real recordings too. Changing to official real evaluation data does not
by itself fix this. We have not established a single cause; threshold suitability,
model/data mismatch and detection behavior still need separate development tests.
The synthetic result (41.10% frame F1) and this real result describe different
corpora, not a same-data model improvement or a calibration experiment.

### Per-class raw frame results

| Class | Precision | Recall | F1 | Positive reference frames |
|---|---:|---:|---:|---:|
| Alarm_bell_ringing | 62.64% | 36.74% | 46.31% | 9464 |
| Blender | 29.67% | 3.46% | 6.20% | 9365 |
| Cat | 82.03% | 39.28% | 53.12% | 5138 |
| Dishes | 94.00% | 0.79% | 1.57% | 5950 |
| Dog | 96.05% | 31.31% | 47.23% | 6601 |
| Electric_shaver_toothbrush | 39.16% | 43.07% | 41.02% | 11669 |
| Frying | 68.20% | 10.83% | 18.69% | 15155 |
| Running_water | 99.24% | 11.57% | 20.73% | 13540 |
| Speech | 92.18% | 58.70% | 71.72% | 24546 |
| Vacuum_cleaner | 100.00% | 7.54% | 14.02% | 15632 |

## Timing and scope

Mac CPU, four threads, after one warm-up: **26.8 ms median**,
95th percentile 37.6 ms per recording. This includes
mel, dB, normalization and network computation, and excludes model/file loading,
resampling, event processing, disk output, waiting for audio and communication.
This is not end-to-end report latency. The BiGRU and normalization see the complete
input window; no deadline-aware controller or Nano execution is tested.

## Known source overlap audit

The public evaluation list was compared by YouTube video ID against the published
2019 weak and unlabeled training lists. Four evaluation recordings share a video
ID with those lists:

- `AaeemnJDijQ_250_260.wav`: weak list has another time segment of the same video.
- `WP8aI1IXNCQ_0_5.wav`: unlabeled list includes the same 0-5 second segment.
- `W7GdMZgJy64_480_490.wav`: unlabeled list has another segment of the same video.
- `aBA1eEsIktM_19_29.wav`: unlabeled list has another segment of the same video.

All-692 scores are preserved. A separate 688-recording sensitivity analysis excludes
these four source IDs. This audit covers published 2019 lists only; actual 2023
training provenance, synthetic source banks and model-selection history have not
been fully audited. **The remaining 688 must not be called proven contamination-free.**

## Reproduce and inspect

Local acquisition and evaluation are isolated from the main project edits:

```bash
.conda/bin/python cache/public_eval_trial/download.py
cache/public_eval_trial/env/bin/python cache/public_eval_trial/evaluate.py
```

The independent scoring environment inherits the existing torch/torchaudio pair
and adds sed_eval; local dependency versions are in
`cache/public_eval_trial/requirements-local.txt`. The official model code revision
is `c6bcb45b8b986ccde5c56bb86eefaf9d19b2320c`. Checkpoint SHA256:
`16a7f4cef26265ce68623bc275a0dac0cdac1d0aa72c1ace4d89ff6c4d5fc4ce`.

- Data, hashes, official annotations and license table: `data/raw/dcase2019_public_eval/`.
Detailed artifacts below are local, ignored by Git, and separate from this
self-contained report.

- Full metrics and protocols: [metrics.json](../output/dcase2019_public_eval/metrics.json).
- Native cached scores and frame lengths: `scores.npz`.
- Decoded event predictions: [events.tsv](../output/dcase2019_public_eval/events.tsv).
- Per-recording counts: [per_recording.tsv](../output/dcase2019_public_eval/per_recording.tsv).

Verification passed: 692-file and 688-file aggregate counts, independent recount
of every frame TP/FP/FN from cached scores and official labels, finite scores,
correct handling of padded cache entries, and complete eleven-second coverage.
A known perfect-prediction example also passed both standard sed_eval scorers.

For later calibration, use a separate development set and freeze its decisions
before evaluating again. This report provides an offline reference; hardware
microphone transfer and the deadline/buffer/communication experiments remain separate.
