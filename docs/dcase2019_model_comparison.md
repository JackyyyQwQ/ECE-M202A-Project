# Official pretrained CNN+BiGRUs on 692 public evaluation recordings

Both official checkpoints were evaluated without training or threshold tuning. The 2019 CNN+BiGRU reproduces its published score. The 2023 CNN+BiGRU, with feature scaling matched to its training code (below), scores higher on every reported metric. This is an offline comparison, not an MCU or deadline experiment.

## Results

All rows use the same 692 recordings and threshold >=0.5. Frame metrics use the same 130,917 intervals. Event metrics use the same onset/offset collars.

| Model | Postprocessing | Frame precision | Frame recall | Frame micro F1 | Event macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 2019 CNN+BiGRU | None | 69.00% | 56.13% | 61.90% | 28.37% |
| 2019 CNN+BiGRU | 5-frame median | 68.98% | 56.06% | 61.85% | 28.97% |
| 2023 CNN+BiGRU | None | 78.06% | 64.90% | 70.87% | 40.52% |
| 2023 CNN+BiGRU | 7-frame median | 77.87% | 64.81% | 70.74% | 39.03% |

Raw rows give the cleanest cross-model comparison; filtered rows use each model's own frame period and smoothing window. F1 is not classification accuracy. Higher recall means fewer missed active frames; event F1 additionally requires matching onset and offset times.

The [official baseline README](https://github.com/turpaultn/DCASE2019_task4/blob/public/baseline/README.md) reports **29.0% event macro F1** on the 2019 public evaluation set with five-frame median filtering. Our **28.97%** is within 0.03 percentage points. This supports a successful baseline reproduction, without claiming an identical historical environment.

## Model and input

- Source: [official 2019 repository](https://github.com/turpaultn/DCASE2019_task4).
- Source revision: `90d35b6191a5d8de05bca816ebc1a836e4a83a00`. Upstream model files were not modified.
- Weights: [author reply and replacement download](https://github.com/turpaultn/DCASE2019_task4/issues/17#issuecomment-1252540486). Checkpoint epoch: **91**; `model` branch, not `model_ema`.
- Checkpoint SHA256: `89694e20d9208fae8e49d44f30c9ec1cb5d2129d476967234351049c4ecb01ac`.
- Architecture: three CNN blocks with GLU, followed by two bidirectional GRU layers with 64 hidden units per direction and ten class outputs. Full-recording future context is available.
- Input: source channels averaged to mono; 44.1 kHz, symmetric Hamming window 2048, hop 511, 64 magnitude mel bins, `htk=False`, `norm=None`. Apply amplitude-to-dB, then checkpoint training mean/standard deviation.
- Output interval: `8 * 511 / 44100` seconds, approximately **92.70 ms**. The 2023 model uses a 16 kHz, 128-mel frontend with 64 ms outputs. Each model keeps its required frontend.
- Short recordings are padded in log-feature space to 864 frames, as in the upstream transforms. Their padded region is not scored. To retain the complete 11-second recording, its features are padded to the next multiple of eight; this produces 119 output frames. Historical code instead truncates at 864 input frames.
- On one 10-second and one 4-second recording, our frontend agrees with the official feature function plus official transforms to within **4.77e-7** after normalization. Modern librosa 0.11.0 is used; this does not prove bit-identical behavior to the original 2019 environment.

## Scoring and checks

Dataset: [official DESED public evaluation archive](https://zenodo.org/records/4560759), 692 recordings and 2,765 events; archive MD5 `c1bf328257523f7e93244f3c1b160eb4`. Both models use all recordings, including the complete 11-second clip. Source channels are averaged to mono. The frame grid starts at [0.45,0.50) seconds, advances by 50 ms, and uses any annotation overlap as a positive target. Each target takes the native prediction bin containing its midpoint. Event metrics use the full available recording and `sed_eval` optimal matching, with 200 ms onset tolerance and offset tolerance max(200 ms, 20% of reference duration).

All 692 audio hashes and official annotations were checked. Independent recounting from cached scores reproduced TP/FP/FN; per-recording totals also agree. All usable scores are finite. No recordings were removed based on performance.

Excluding the four previously identified training-video overlaps leaves 688 recordings: raw frame F1 **61.88%**, event macro F1 **28.29%**. This only checks the published 2019 weak/unlabeled lists; it does not establish complete independence from training or model selection.

CPU computation for the 2019 pipeline: median **32.12 ms**, p95 **41.28 ms**, four PyTorch threads. This covers mel extraction, normalization and network inference; it excludes file loading, decoding predictions, communication and waiting for audio. It is not online latency. The timing runs were not a controlled speed comparison between model years.

## 2023 model: feature scaling

The 2023 checkpoint (`legacy_epoch=176-step=20886.ckpt`, [Zenodo 7759146](https://zenodo.org/records/7759146)) was trained with the March 2023 upstream `TorchScaler` (commit `9f9b610`), whose per-clip minmax maps log-mel features to **[0, 1]**. The pinned model-code checkout (`c6bcb45`, June 2024) later changed that scaler to return [-1, 1]. Weights load strictly under either range, so the mismatch raises no error; it shifts every input and suppresses detections, most strongly for stationary classes. `scripts/pretrained_sed.py` now uses the 2023 range and the recipe's `take_log` (amin 1e-5, clamp [-50, 80] dB). The mel frontend, class order and CRNN frame-output path were checked to be identical between the two code versions.

With the [-1, 1] input, the same checkpoint and protocol gave 39.20% raw frame F1 and 16.55% event macro F1, with most stationary-class events missed; after the correction it gives 70.87% and 40.52%. The 40.52% is consistent in magnitude with the official 40.7% collar F1 on the 2023 dev-test set (a different set, so not a reproduction claim). Earlier 2023 scores (the 64-recording pilot and the 2,045-recording synthetic run) used the old scaling and are superseded. A NumPy reimplementation of the checkpoint (local, `cache/numpy_crnn_check/`) independently gave 0.711 frame F1 on every third public file (native 64 ms grid).

Per-class raw frame F1 after the correction ranges from 45% (dishes) to 89% (speech); running water (58%), frying and blender (62%) remain the weakest stationary classes. Excluding the four known training-video overlaps gives 70.73% frame F1 and 40.39% event macro F1. CPU computation: median 25.9 ms, p95 32.2 ms per recording (same scope as the 2019 timing below).

## Implication for our project

The corrected 2023 checkpoint is the stronger offline reference on this set, and its 16 kHz frontend matches the planned Nano capture rate; the 2019 model would require 44.1 kHz input. We therefore use the 2023 model as the reference detector, with the 2019 model as a reproduced cross-check. It does not yet provide a causal baseline, bounded lookahead, a learned MCU encoder or refinement. These results cannot be credited to our controller. Freeze this test result; tune future thresholds or models on a separate training/validation split.

## Evidence and reproduction

The [committed result summary](../results/public_eval_summary.json) contains both checkpoints, scores and scoring definitions. Detailed per-run artifacts (`metrics.json`, `scores.npz`, `events.tsv`, `per_recording.tsv`) are written to `output/public_eval/<model>/`, which is ignored by Git.

Setup, from the repository root (audio, checkpoints and upstream code stay in ignored folders):

```sh
conda env create --prefix ./.conda --file environment.yml   # once
conda activate "$PWD/.conda"
python scripts/fetch_public_eval.py                          # 692 WAVs, MD5/SHA-256 checked

# 2023 model: upstream code and checkpoint archive (MD5 65e89bb452e141267e41e5c849e05ac8)
git clone https://github.com/DCASE-REPO/DESED_task.git cache/DESED_task
git -C cache/DESED_task checkout c6bcb45b8b986ccde5c56bb86eefaf9d19b2320c
mkdir -p checkpoints/dcase2023
curl -L --fail -o checkpoints/dcase2023/baseline.zip https://zenodo.org/records/7759146/files/dcase2023_task4_baseline.zip
unzip -j checkpoints/dcase2023/baseline.zip 'ckpt/legacy_epoch=176-step=20886.ckpt' -d checkpoints/dcase2023

# 2019 model: upstream code; weights from the author's reply in upstream issue #17
git clone https://github.com/turpaultn/DCASE2019_task4.git cache/DCASE2019_task4
git -C cache/DCASE2019_task4 checkout 90d35b6191a5d8de05bca816ebc1a836e4a83a00
mkdir -p checkpoints/dcase2019   # place dcase2019_t4_mdl here
```

The 2023 runner checks the upstream revision and a clean checkout; the 2019 loader checks its revision. Checkpoint SHA-256 values are recorded in the summary and in each `metrics.json`. Evaluate and then independently recount each model:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python scripts/evaluate_public.py --model 2023
python scripts/verify_public.py --model 2023
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python scripts/evaluate_public.py --model 2019
python scripts/verify_public.py --model 2019
```

`verify_public.py` needs no model weights: it rebuilds the 50 ms targets, recounts raw TP/FP/FN from cached scores and checks per-recording totals.
