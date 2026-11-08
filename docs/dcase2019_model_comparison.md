# DCASE 2019 pretrained model: comparison on 692 public evaluation recordings

The author-provided 2019 CNN+BiGRU checkpoint was evaluated without training or threshold tuning. It performs better than our previously tested 2023 checkpoint on this dataset. This is an offline comparison, not an MCU or deadline experiment.

## Results

All rows use the same 692 recordings and threshold >=0.5. Frame metrics use the same 130,917 intervals. Event metrics use the same onset/offset collars.

| Model | Postprocessing | Frame precision | Frame recall | Frame micro F1 | Event macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 2019 CNN+BiGRU | None | 69.00% | 56.13% | 61.90% | 28.37% |
| 2023 CNN+BiGRU | None | 70.67% | 27.13% | 39.20% | 16.55% |
| 2019 CNN+BiGRU | 5-frame median | 68.98% | 56.06% | 61.85% | 28.97% |
| 2023 CNN+BiGRU | 7-frame median | 70.38% | 26.63% | 38.64% | 17.16% |

The raw rows provide the cleanest comparison. The filtered rows use different native frame periods and smoothing windows and should be read as separate configurations. F1 is not classification accuracy. Higher recall means fewer missed active frames; event F1 additionally requires matching onset and offset times.

The [official baseline README](https://github.com/turpaultn/DCASE2019_task4/blob/public/baseline/README.md) reports **29.0% event macro F1** on the 2019 public evaluation set with five-frame median filtering. Our **28.97%** is within 0.03 percentage points. This supports a successful baseline reproduction, without claiming an identical historical environment.

## Model and input

- Source: [official 2019 repository](https://github.com/turpaultn/DCASE2019_task4).
- Source revision: `90d35b6191a5d8de05bca816ebc1a836e4a83a00`. Upstream model files were not modified.
- Weights: [author reply and replacement download](https://github.com/turpaultn/DCASE2019_task4/issues/17#issuecomment-1252540486). Checkpoint epoch: **91**; `model` branch, not `model_ema`.
- Checkpoint SHA256: `89694e20d9208fae8e49d44f30c9ec1cb5d2129d476967234351049c4ecb01ac`.
- Architecture: three CNN blocks with GLU, followed by two bidirectional GRU layers with 64 hidden units per direction and ten class outputs. Full-recording future context is available.
- Input: source channels averaged to mono; 44.1 kHz, symmetric Hamming window 2048, hop 511, 64 magnitude mel bins, `htk=False`, `norm=None`. Apply amplitude-to-dB, then checkpoint training mean/standard deviation.
- Output interval: `8 * 511 / 44100` seconds, approximately **92.70 ms**. The 2023 model uses its own 16 kHz frontend and 64 ms outputs. Each model keeps its required frontend.
- Short recordings are padded in log-feature space to 864 frames, as in the upstream transforms. Their padded region is not scored. To retain the complete 11-second recording, its features are padded to the next multiple of eight; this produces 119 output frames. Historical code instead truncates at 864 input frames.
- On one 10-second and one 4-second recording, our frontend agrees with the official feature function plus official transforms to within **4.77e-7** after normalization. Modern librosa 0.11.0 is used; this does not prove bit-identical behavior to the original 2019 environment.

## Scoring and checks

Dataset: [official DESED public evaluation archive](https://zenodo.org/records/4560759), 692 recordings and 2,765 events; archive MD5 `c1bf328257523f7e93244f3c1b160eb4`. Both models use all recordings, including the complete 11-second clip. Source channels are averaged to mono. The frame grid starts at [0.45,0.50) seconds, advances by 50 ms, and uses any annotation overlap as a positive target. Each target takes the native prediction bin containing its midpoint. Event metrics use the full available recording and `sed_eval` optimal matching, with 200 ms onset tolerance and offset tolerance max(200 ms, 20% of reference duration).

All 692 audio hashes and official annotations were checked. Independent recounting from cached scores reproduced TP/FP/FN; per-recording totals also agree. All usable scores are finite. No recordings were removed based on performance.

Excluding the four previously identified training-video overlaps leaves 688 recordings: raw frame F1 **61.88%**, event macro F1 **28.29%**. This only checks the published 2019 weak/unlabeled lists; it does not establish complete independence from training or model selection.

CPU computation for the 2019 pipeline: median **32.12 ms**, p95 **41.28 ms**, four PyTorch threads. This covers mel extraction, normalization and network inference; it excludes file loading, decoding predictions, communication and waiting for audio. It is not online latency. The timing runs were not a controlled speed comparison between model years.

## Implication for our project

This checkpoint is a better starting candidate for reliable offline detections on the 2019 public evaluation data. It does not yet provide a causal baseline, bounded lookahead, a learned MCU encoder or refinement. These results cannot be credited to our controller. Freeze this test result; tune future thresholds or models on a separate training/validation split.

## Evidence and local reproduction

[Committed result summary](../results/public_eval_summary.json) contains both checkpoints, scores and scoring definitions. The following detailed artifacts are local and ignored by Git:

- [Metrics and provenance](../output/dcase2019_model_public_eval/metrics.json)
- [Predicted events](../output/dcase2019_model_public_eval/events.tsv)
- [Per-recording counts](../output/dcase2019_model_public_eval/per_recording.tsv)

The following commands run from the project root using the existing isolated evaluation environment. Cached code, weights, audio and detailed outputs are local ignored artifacts; these commands assume they are present. No dependencies were installed into the main project environment.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 cache/public_eval_trial/env/bin/python cache/dcase2019_trial/evaluate.py
cache/public_eval_trial/env/bin/python cache/dcase2019_trial/verify.py
```
