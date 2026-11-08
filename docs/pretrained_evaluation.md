# Pretrained BiGRU on DCASE2019: fixed 64-recording pilot

This page preserves the 64-file pilot. See the subsequent [full-corpus report](pretrained_full_report.md) for all 2,045 recordings.

Measured November 7, 2026. Frozen official DCASE 2023 plain CNN+BiGRU student;
no training, threshold tuning or checkpoint selection on this pilot.

## Data and reproducibility

Source: [DCASE2019 synthetic dataset, Zenodo 2583796](https://zenodo.org/records/2583796).
The ZIP directory contains **2,045 WAV files** and a tab-separated onset/offset
annotation file. We selected 64 filenames before inference with Python 3.11
`random.Random(2026).sample(sorted_audio_names, 64)`, then sorted the selection.
All ten official classes occur in the selected annotations.

This is a **new pilot subset**, not the original 32-recording validation split.
No training or validation manifest from the earlier experiment was reconstructed.
Whether this data overlaps the pretrained model's training material has not been
audited, so this is not an independent generalization benchmark.

The downloader reads ZIP byte ranges and verifies each selected member's size and
CRC32. Full-archive MD5 was not checked because the entire archive was not fetched.
The manifest records member names, CRC32 and extracted WAV SHA256 hashes. The
complete annotation file is retained locally with its hash. Five WAVs contain one
extra sample at 44.1 kHz; that final sample was trimmed before 16 kHz resampling.
No files were excluded after inspecting prediction quality. The evaluation covers
640 seconds of audio and **12,224 scored 50 ms intervals**.

With the [pretrained environment and weights](pretrained_baseline.md) already set up:

```bash
python scripts/fetch_dcase_pilot.py --count 64 --seed 2026
python -m scripts.evaluate_pretrained
```

Local audio remains under ignored `data/raw/dcase2019/`; detailed scores, per-file
counts and predictions are in ignored `output/pretrained_pilot/`. Small evidence:
[metrics](../results/pretrained_pilot_metrics.json),
[fixed manifest](../results/pretrained_pilot_manifest.csv),
[per-class counts](../results/pretrained_pilot_per_class.tsv).

## Evaluation contract

- Full ten-second input, official frontend and pretrained student weights. No MCU,
  bandwidth limit, finite lookahead or reporting deadline is evaluated.
- Official output uses 64 ms bins. For comparison-oriented frame scoring, assign
  the bin containing each 50 ms target's midpoint. Targets are [0.45, 0.50), ...,
  [9.95, 10.00), matching the old loader's 191-interval coverage. A class is positive
  if its annotation overlaps the target interval. No startup padding targets.
- **Primary:** raw probabilities, threshold >=0.5, no smoothing. **Secondary:**
  symmetric 7-frame median filtering at the same threshold. Report both;
  do not pick a threshold based on these results.
- Frame micro precision/recall/F1 pool all frame/class decisions. Macro F1 is the
  unweighted mean of ten class F1 scores, with undefined ratios set to zero.
- Events are consecutive positive 50 ms intervals. Match within class/recording
  using positive overlap, maximum pair count, then total IoU. This permissive
  overlap F1 is **not** collar-based F1, PSDS or an official DCASE benchmark score.
- Boundary MAE uses these matched pairs only. Exclude onset timing errors for
  references starting <=0.45 s and offset errors for references ending >=10 s.
  Unmatched events remain false/missed events in event F1.

## Results

| Metric | Raw, primary | Median-7, secondary |
|---|---:|---:|
| Frame micro precision | 82.74% | 82.89% |
| Frame micro recall | 26.13% | 25.57% |
| **Frame micro F1** | **39.72%** | **39.08%** |
| Frame macro F1 | 26.76% | 26.38% |
| Permissive event overlap F1 | 53.37% | 52.20% |
| Matched events / predicted / reference | 95 / 138 / 218 | 83 / 100 / 218 |
| Matched onset MAE | 296.7 ms (91) | 269.5 ms (80) |
| Matched offset MAE | 370.1 ms (79) | 344.4 ms (70) |

Raw frame counts: **1,985 TP, 414 FP, 5,611 FN**. High precision coexists with low
recall; this model misses many active frames at the untuned threshold.

### Raw results by class

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Alarm_bell_ringing | 51.05% | 52.80% | 51.91% |
| Blender | 0.00% | 0.00% | 0.00% |
| Cat | 95.83% | 29.24% | 44.81% |
| Dishes | 0.00% | 0.00% | 0.00% |
| Dog | 100.00% | 9.34% | 17.09% |
| Electric_shaver_toothbrush | 85.19% | 46.86% | 60.46% |
| Frying | 0.00% | 0.00% | 0.00% |
| Running_water | 0.00% | 0.00% | 0.00% |
| Speech | 97.24% | 48.49% | 64.71% |
| Vacuum_cleaner | 100.00% | 16.70% | 28.61% |

Four classes have zero F1 despite positive reference examples. We have not
established whether thresholds, domain mismatch or another input/model issue is
the main cause; pretrained weights do not guarantee an adequate project detector.
Do not treat these four classes as absent from the data.

Mel + dB + normalization + network computation on Mac CPU, four threads, after one
warm-up: **23.6 ms median**, range 22.3–28.3 ms across 64 recordings. Timing excludes
file/model loading, resampling, event processing, I/O, audio collection and transport.
Full-window BiGRU and normalization use future audio. These are computation times,
not streaming report latency. They are separate from the earlier 30.0 ms smoke run.

## Relation to the old custom model

The old report recorded 21.39% frame F1 for 32D and 23.29% for 128D, on a different
32-recording validation subset. The new 39.72% result is encouraging as an initial
reference, but is **not a measured improvement on the same examples**. Training
resources, backbone, future context, preprocessing and the evaluated recordings
differ. The old custom weights and split list are still missing.

Keep this pilot unchanged. Next audit the frontend and failure classes, then use a
separate development subset for threshold decisions and fresh evaluation recordings.
When the custom weights are available, run both systems on the saved 64-file manifest
with the same evaluation code. Only then quantify a same-data system comparison;
a directionality ablation additionally needs matched training and model capacity.


## Follow-up diagnosis: default threshold misses positive frames

The official class order matches our mapping. On three pilot recordings, our
normalized mel output was compared against the pinned upstream `take_log` and
`TorchScaler` implementations: maximum absolute difference was zero in all three.
This checks these frontend operations, not every possible dataset/domain issue.

For four zero-F1 classes, the highest raw score on any positive evaluation frame
was below 0.5: Blender 0.2394, Dishes 0.06594, Frying 0.2826, Running_water 0.4285.
Thus no positive frame of these classes passes the current threshold. Dishes still
has seven false-positive frames elsewhere. These are scores, not calibrated
probabilities or class accuracies.

Ranking is not uniformly absent: Frying frame ROC AUC is 0.9394; Dishes is 0.7312
on this pilot. This suggests threshold suitability contributes to the failure,
while class separation also varies. AUC does not establish useful precision,
boundary quality or a successful threshold fix. Domain shift remains a hypothesis;
we have not established its cause or measured a correction.

See [diagnostic score distributions](../results/pretrained_pilot_diagnosis.json).
No threshold was selected and the original 39.72% result is unchanged. Use a
separate development subset for per-class threshold calibration, followed by fresh
evaluation recordings, rather than reporting a threshold optimized on this pilot.
