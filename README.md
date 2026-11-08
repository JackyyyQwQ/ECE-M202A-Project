# Deadline-Aware Collaborative Audio Inference

**P13 · UCLA ECE M202A / CS M213A · Fall 2026**

Junchi Chen · Wei Zhang

[Project website](https://jackyyyqwq.github.io/ECE-M202A-Project/) ·
[Proposal](docs/proposal.md) · [Architecture](docs/architecture.md) ·
[Feasibility evidence](docs/feasibility.md)

## What we are building

A Nano 33 BLE Sense running Zephyr sends compact learned audio features to a
laptop. The laptop detects sound events and estimates their onset/offset times.
After a causal prediction, the proposed receiver can report immediately, wait
for bounded future features and run a BiGRU, or request buffered PCM for refinement.
We will compare these paths before designing the action selector.

**Research question:** when is it worth waiting for future features versus
retrieving expiring historical audio, under reporting deadlines and communication
budgets? Adaptive selection is a hypothesis to test, not a guaranteed improvement.

## Current status

| Component | Status |
|---|---|
| Window/sequence loaders; CNN + causal GRU | Implemented; offline feasibility results reported |
| 200 ms audio refiner and context-only control | Implemented; offline comparison reported |
| Data preparation, training/evaluation runners, checkpoints | Not included; recover from original experiments |
| Project Zephyr buffer/replay, deployed encoder, transport | Planned |
| Official pretrained CNN+BiGRU | Both evaluated on 692 real recordings; corrected 2023 reaches 70.87% frame F1 and 40.52% event F1; 2019 reproduces its published 29.0% |
| Bounded-lookahead adaptation, online receiver and controller | Planned; compare paths before selection |

The original November 2 experiment used 128 training and 32 validation recordings
for the baseline. Across 117 eligible matched boundaries, MAE was **380.2 ms**
initially, **373.6 ms** with audio + context, and **372.4 ms** with context only.
Audio refinement has **not** shown an overall benefit over context-only, and
event detection did not improve. See the [full report](docs/feasibility.md) for
matching rules, false/missed events, byte costs, sample sizes and limitations.
These are preserved preliminary results, not a new run after this reorganization.

## Pretrained baseline progress

The official **2019 and 2023 CNN+BiGRUs** run offline on Mac using the authors'
pretrained weights. All **692 real DCASE2019 public evaluation recordings** were
scored without training or threshold tuning:

| Frozen model, raw output | Frame precision | Frame recall | Frame micro F1 | Event macro F1 |
|---|---:|---:|---:|---:|
| 2019 CNN+BiGRU | 69.00% | 56.13% | 61.90% | 28.37% |
| 2023 CNN+BiGRU | 78.06% | 64.90% | **70.87%** | **40.52%** |

With the official five-frame median filter, the 2019 model reaches **28.97% event
macro F1**, close to the published **29.0%**. Frontend checks and independent score
recounting passed. This establishes a working offline reference; it does not yet
implement our MCU encoder, bounded lookahead, refinement or controller.

Both models use their own CNN/mel frontend and full future context. They do not
accept our custom 32D features. The 2019 frontend requires 44.1 kHz; the 2023
frontend requires 16 kHz. Do not directly move the official CNN onto the Nano.
See the [comparison report](docs/dcase2019_model_comparison.md) for sources,
scoring, reproduction commands and limitations, and the
[committed score summary](results/public_eval_summary.json) for evidence.

The 2023 checkpoint expects per-clip log-mel features scaled to [0, 1], as in its
2023 training code; the pinned 2024 upstream code scales to [-1, 1], which had
suppressed detections (39.20% frame F1) in our earlier run. With the training-code
range it is the stronger model, and its 16 kHz frontend matches the Nano capture
plan, so it is our reference detector. See the
[comparison report](docs/dcase2019_model_comparison.md#2023-model-feature-scaling).

The existing 2023 inference entry point remains available with the project conda
environment active and a local recording of at most ten seconds:

```bash
python scripts/pretrained_sed.py /path/to/recording.wav --output output/pretrained_smoke
```

Next, establish a reproducible causal baseline on separate training/validation
recordings, then compare bounded future context and PCM refinement. Keep the
public evaluation set fixed; do not tune thresholds or the controller on it.

## Repository map

```text
models/       CNN/GRU and boundary refiner (implemented)
data/         Dataset loaders and source/preparation notes
firmware/     Planned Zephyr device application
receiver/     Planned Python laptop runtime and controller
scripts/      Pretrained inference; original training runners still pending
configs/      Settings to add alongside working runners
tests/        Data/model correctness and pretrained timestamp export checks
results/      Original small feasibility plots
docs/         One-page architecture and full feasibility report
index.html    GitHub Pages website
```

## Python environment

From the repository root, use a separate conda environment for offline models.
Create it once (the local diagnostic setup already has this environment):

```bash
conda env create --prefix ./.conda --file environment.yml
conda activate "$PWD/.conda"
```

On later visits, only run the activation command. The environment uses Python 3.11
and NumPy 1.26 to avoid the observed base-environment binary incompatibility.
It is a diagnostic environment, not the original experiment's version lock.
`requirements.txt` lists the direct model/loader dependencies. Check imports:

```bash
python -c "from models import CausalActivityModel; print(CausalActivityModel())"
```

Run the synthetic baseline checks without downloading audio:

```bash
python -m unittest discover -s tests -p 'test_baseline.py' -v
```

These check data alignment, causality and numerical behavior, not detection
accuracy. Reproducing the old custom-model detection scores still requires its
original training artifacts and split manifests.

See [data setup](data/README.md) for the local WAV, annotation, split and class
files that must be supplied. Once they are present:

```bash
python -m data.audio_dataset
```

This inspects dataset tensors; it does not train or reproduce reported metrics.
The dataset is [DCASE2019 task 4 synthetic data](https://zenodo.org/records/2583796).
The team's preparation summary reports 2,045 recordings and an 80/10/10
recording-level split with seed 2026; preserve the actual original manifests.

## Simulation and hardware

There are three development modes:

1. **Offline Python:** recorded audio -> encoder -> detector/refiner. Current code.
2. **Zephyr native_sim + Python receiver:** planned paced replay, finite buffer
   and request/response integration. The laptop runs ordinary Python, not Zephyr.
3. **Nano + Zephyr + Python receiver:** planned microphone/replay and USB runtime.

The [course Zephyr repo](https://github.com/UCLA-ECEM202A/software/tree/83a3ac6ccf585b2b647d3d6c654b616de8c29cb8/zephyr)
provides the toolchain and reference apps. In that separate course folder,
`bash ./zephyr setup` builds its environment and `bash ./zephyr sim queue-demo`
runs a synthetic queue test. `bash ./zephyr build-usb acoustic-event` builds
Sense Rev2 microphone firmware; this app cannot run in native_sim.

On Junchi's current machine, an existing sibling `audio-lab/` launcher uses an
isolated image/cache. From this checkout, `bash ../audio-lab/run sim queue-demo`
runs that course demo. This helper is local setup, not a dependency shipped by
this repository. Smoke/queue runs and an acoustic cross-build passed there;
they do not establish a project pipeline or a physical board run. Native_sim
does not measure Nano processor speed, BLE performance, or energy.

## Next work and evaluation

1. Recover artifacts and reproduce/improve the causal CNN + GRU baseline.
2. Compare causal GRU, bounded-lookahead BiGRU, context-only correction and
   PCM refinement with a fixed encoder, matched reporting budgets and explicit
   byte costs. Separate capacity effects from future-context effects.
3. Implement finite PCM history and sample-index requests; connect the receiver.
4. Measure selection headroom, then test a simple REPORT / LOOKAHEAD / REFINE
   controller. The first version chooses one action per candidate. If one fixed
   path dominates, simplify and report that result.

The controller only sees initial predictions and currently available resource
information. Full-clip BiGRU results and post-hoc best-path choices are offline
references, not deployable policies. Include waiting, transport, queueing and
inference in latency. Define deadline anchors and clock mapping before experiments.

Measure timely event quality, boundary error, false/missed events, bytes, latency,
deadline misses and MCU resources. Stress-test link conditions and buffer lifetime.
Minimum target: Nano/laptop USB prototype; paced replay is the microphone fallback.
See the [revised proposal](docs/proposal.md) for scope, controls and responsibilities.

## Development and website

Keep code small and readable, and keep
[docs/architecture.md](docs/architecture.md) to one page as the system changes.
Do not check in large audio, caches, virtualenvs or model checkpoints.

The website is plain HTML/CSS with no build step. Preview from the repository root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/`; stop with Ctrl-C. Update `index.html` alongside
new evidence. GitHub Pages serves the root website when configured to publish
this branch. No checkpoint PDF is included yet; add its direct link when ready.

## References

See [related work and proposed adaptations](docs/references.md) for summaries,
differences, and experiments for each paper.

- [Compression at the Edge for Energy- and Bandwidth-Efficient Industrial Audio Analysis](https://doi.org/10.1016/j.procs.2025.07.180)
- [Communication-Efficient Multi-Modal Edge Inference via Uncertainty-Aware Distributed Learning](https://arxiv.org/abs/2601.14942)
- [Wireless Evidence Acquisition for Multimodal Inference Constrained by Temporal Windows](https://arxiv.org/abs/2609.31428)
- [Sound Event Bounding Boxes](https://arxiv.org/abs/2406.04212)
- [Convolutional Recurrent Neural Networks for Polyphonic Sound Event Detection](https://arxiv.org/abs/1702.06286)
- [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)

We build on feature compression and selective evidence acquisition, focusing on
temporal audio segments, finite MCU history and on-time boundary estimates.
