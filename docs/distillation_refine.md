# Compact encoder: distillation, int8 TFLM and REFINE simulation

November 11, 2026. **Preliminary, offline simulation on the development
(validation) set.** No training of the BiGRU; the public evaluation set was not used.

## Summary

- A depthwise-separable CNN (DS-CNN) distilled from the reference 2023 CNN keeps
  98.8% of reference event micro F1 (0.417 vs 0.422) with 7x fewer convolution MACs
  (9.62 vs 70.1 M MAC/s). A 56x cheaper student (1.26 M MAC/s) drops to 0.390.
- For that small student, fetching PCM around the most uncertain frames and
  recomputing them with the reference CNN recovers about 1/3 of the gap at
  +2 kB/s uplink (5.5% of audio) and about 70% at +8 kB/s (24%). Uncertainty
  ranking beats random selection with the same budget in every setting and is on
  par with an oracle ranking.
- The larger student without REFINE (0.417 at 2 kB/s) beats the small student with
  REFINE at 10 kB/s (0.414): REFINE trades uplink bytes for MCU compute.
- The small student in int8, run per frame, keeps its event F1 (0.392 vs 0.393
  float) and runs in TensorFlow Lite Micro under Zephyr: 25.5 KB model, 48 KB
  tensor arena (measured in native_sim).
- Online use under a deadline is covered by the [controller simulation](controller.md).

## Setup

**Teacher:** the CNN part of the official DCASE 2023 CRNN
(`legacy_epoch=176-step=20886.ckpt`), frozen. **BiGRU and heads:** official,
frozen, unchanged. The split teacher (CNN, then BiGRU) is checked against the
official forward pass (max |diff| < 1e-5).

**Student** (`scripts/distill_cnn.py`): a 3x3 stem (frequency stride 2), five
depthwise-separable blocks (depthwise 3x3, average pooling, pointwise 1x1) and a
linear 1x1 projection to 128 channels. Time pooling matches the teacher (2, then 2),
so it outputs one 128-D feature per 64 ms frame. `--width` scales the channels
(1.0 = 32/64/128; 0.25 = 8/16/32), as a MobileNets width multiplier.

**Training:** 2,045 DCASE2019 synthetic clips (audio only, no labels), 10 s each,
5% held out for the training curve. Loss: MSE between student and teacher
features, each channel standardised. AdamW, one-cycle LR 3e-3, 40 epochs, batch 32,
seed 2026, Apple MPS. Same frontend as the teacher, including per-clip [0, 1]
normalisation.

**Evaluation:** 916 validation clips (development set), whole-clip context, the
same scoring as the [lookahead sweep](lookahead_sweep.md): event F1 with onset
collar 200 ms and offset max(200 ms, 20%); hysteresis 0.5/0.3 unless stated.

## Distillation results

| Encoder | Parameters | Conv M MAC/s | Hold-out MSE | Frame F1 | Event F1 micro | Event F1 macro |
|---|---:|---:|---:|---:|---:|---:|
| Teacher (reference CNN) | — | 70.1 | — | 0.709 | 0.422 | 0.405 |
| Student, width 1.0 (`ds_mse`) | 69,568 | 9.62 | 0.052 | 0.695 | 0.417 | 0.391 |
| Student, width 0.25 (`ds_w025`) | 8,656 | 1.26 | 0.149 | 0.632 | 0.390 | 0.339 |

All three post-processing rules give the same ordering; see
[`results/distill/`](../results/distill/). MACs count convolution layers only.

## REFINE simulation

`scripts/refine_sim.py`. The Nano sends student features (128-D int8 every 64 ms,
2 kB/s). Each clip may also fetch PCM (32 kB/s) up to a budget of extra kB/s.
Frames are ranked, and each selected frame is expanded to a chunk of +-w seconds
(shrunk to the remaining budget, never below +-0.128 s). The laptop runs the
reference CNN on each fetched chunk alone (zero padding at its edges), replaces
the features of every frame in the chunk and reruns the frozen BiGRU.

Rankings, each spending the same budget:
**uncertainty** (frames whose most uncertain class is closest to p = 0.5; online-
feasible), **random** (mean of 3 seeds; control) and **oracle** (largest
student/teacher output difference; not online-feasible).

Width 0.25 student, event F1 micro:

| Extra budget | PCM fetched | Uncertainty | Oracle | Random (mean of 3) |
|---|---:|---:|---:|---:|
| none (student only) | 0% | 0.390 | — | — |
| +2 kB/s | 5.5% | 0.401 | 0.399 | 0.394 |
| +4 kB/s, +-0.5 s | 11% | 0.405 | 0.406 | 0.395 |
| +8 kB/s, +-0.25 s | 24% | 0.414 | 0.412 | 0.395 |
| all PCM | 100% | 0.425 | — | — |

At +1 kB/s nothing fits (budget 4 frames < 5-frame minimum chunk); at +2 kB/s all
chunk sizes reduce to the same 9 frames. Best macro F1: 0.339 -> 0.362.
"All PCM" (0.425) is slightly above the teacher (0.422) because chunk recomputation
drops the last partial 4-frame mel group. Full table and figure:
[`results/distill/ds_w025_refine_summary.json`](../results/distill/ds_w025_refine_summary.json),
[`ds_w025_refine_curve.png`](../results/distill/ds_w025_refine_curve.png).

An earlier version refined only the selected frame while charging 0.86 s of PCM for
its receptive field; it recovered little and matched random selection. Recomputing
whole fetched chunks is what makes REFINE effective.

## int8 TFLite and TensorFlow Lite Micro

`scripts/export_tflite.py` rebuilds the student with TensorFlow ops (BatchNorm folded;
max |diff| vs PyTorch 4.8e-6), quantises it to full int8 with 500 windows of the
synthetic training audio, and writes `firmware/tflm_probe/src/model_<run>.h`.

**Per-frame inference.** Each call sees 40 mel frames (0.64 s), covering the student's
36-frame receptive field, and keeps one 64 ms output frame (mel frames 16-19 of the
window, so 20 future mel frames = 0.32 s look-ahead). Windows start on multiples of 4
mel frames, so interior frames equal the whole-clip student. Cost: 0.86 M MAC per call,
about 13.5 M MAC/s, roughly 10x the whole-clip cost because windows overlap.

| Width 0.25, 916 validation clips, hysteresis | Frame F1 | Event F1 micro | Event F1 macro |
|---|---:|---:|---:|
| float, whole clip | 0.632 | 0.390 | 0.339 |
| float, per frame | 0.632 | 0.393 | 0.343 |
| int8 TFLite, per frame | 0.632 | 0.392 | 0.337 |

Differences are within noise. Ops: PAD, CONV_2D, DEPTHWISE_CONV_2D, AVERAGE_POOL_2D,
MEAN. Details: [`results/distill/ds_w025_tflite_eval.json`](../results/distill/ds_w025_tflite_eval.json).

**TFLM probe (Zephyr native_sim).** `firmware/tflm_probe` loads the model, allocates
the arena and runs one inference: model 25,504 bytes, arena used 48,848 bytes,
input int8 1x40x128x1, output int8 1x10x1x128, invoke ok. Not yet measured: the Nano
build's FLASH/RAM report (`bash firmware/run probe-nano ds_w025 64`) and on-device
latency. Python TFLite and TFLM may differ by +-1 LSB on some outputs; a golden-vector
test is not yet in place.

## Limitations

- **Normalisation:** student, teacher and REFINE chunks use whole-clip min/max
  scaling, which an online system cannot know. Optimistic.
- **Offline:** the BiGRU sees the whole clip; no deadline, lookahead, buffer expiry,
  round-trip delay or request overhead. Online, a +-w chunk around frame t is past
  audio once the system has waited L >= w, and still buffered if L + w <= 2 s.
- **Budget:** allocated per clip with the whole clip in view.
- **MCU:** int8 per-frame inference is checked in Python and in native_sim TFLM;
  the Nano memory report, on-device latency and the mel frontend on the Nano are
  untested.
- **Statistics:** one training run per width, no confidence intervals; event F1
  differences below about 0.01 are within noise. Settings were chosen on the
  development set.

## Reproduce

Data: `python scripts/fetch_dcase_pilot.py --all` (training audio) and the
validation set (`scripts/fetch_validation.py`). Trained students are committed in
`models/distill/<run>/` (`student.pt`, `train.json`), so training is optional.

```bash
python scripts/distill_cnn.py cache                              # teacher targets, once
python scripts/distill_cnn.py train --run ds_w025 --width 0.25   # optional: retrain
python scripts/distill_cnn.py eval --run ds_w025                 # uses the committed weights if not retrained
python scripts/refine_sim.py --run ds_w025                       # about 9 min
python scripts/export_tflite.py --run ds_w025                    # int8 model, three F1s, C header
bash firmware/run probe ds_w025                                  # TFLM size probe in native_sim
```

Committed weights (SHA-256): `ds_mse` 14acdd54ef41c94effd18cc085c964de52ff9eaca753b3e3fa1561609308f610;
`ds_w025` e94edca76a2b9fed4308c5701f0255eca8524a06344f84e6a8c3f3acd97734cf.

## Next

Train width 0.5; train the student with an online-feasible normalisation; Nano
FLASH/RAM report and a golden-vector check; measure BLE throughput. REFINE together
with lookahead and a deadline is now simulated in the [controller](controller.md).
