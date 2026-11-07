# Deadline-Aware Collaborative Audio Inference

**P13 · UCLA ECE M202A / CS M213A · Fall 2026**

Junchi Chen · Wei Zhang

[Project website](https://jackyyyqwq.github.io/ECE-M202A-Project/) ·
[Architecture](docs/architecture.md) · [Feasibility evidence](docs/feasibility.md)

## What we are building

A Nano 33 BLE Sense running Zephyr sends compact learned audio features to a
laptop. The laptop detects sound events and estimates their onset/offset times.
When extra evidence may help, it requests a short original-audio segment from
the MCU's finite history buffer and refines the estimate.

**Research question:** how should a receiver choose useful audio segments before
they are overwritten, while staying within a communication budget and reporting
deadline? The contribution is the request policy and its systems evaluation.

## Current status

| Component | Status |
|---|---|
| Window/sequence loaders; CNN + causal GRU | Implemented; offline feasibility results reported |
| 200 ms audio refiner and context-only control | Implemented; offline comparison reported |
| Data preparation, training/evaluation runners, checkpoints | Not included; recover from original experiments |
| Project Zephyr buffer/replay, deployed encoder, transport | Planned |
| Online laptop receiver and deadline-aware controller | Planned |

The original November 2 experiment used 128 training and 32 validation recordings
for the baseline. Across 117 eligible matched boundaries, MAE was **380.2 ms**
initially, **373.6 ms** with audio + context, and **372.4 ms** with context only.
Audio refinement has **not** shown an overall benefit over context-only, and
event detection did not improve. See the [full report](docs/feasibility.md) for
matching rules, false/missed events, byte costs, sample sizes and limitations.
These are preserved preliminary results, not a new run after this reorganization.

## Repository map

```text
models/       CNN/GRU and boundary refiner (implemented)
data/         Dataset loaders and source/preparation notes
firmware/     Planned Zephyr device application
receiver/     Planned Python laptop runtime and controller
scripts/      Space for original/recovered experiment runners
configs/      Settings to add alongside working runners
tests/        Planned focused buffer, protocol and causality checks
results/      Original small feasibility plots
docs/         One-page architecture and full feasibility report
index.html    GitHub Pages website
```

## Python environment

From the repository root, use a separate environment for offline models:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency list covers current model/loaders; it is not the original
experiment's version lock. Check that the definitions load:

```bash
python -c "from models import CausalActivityModel; print(CausalActivityModel())"
```

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

- Recover data preparation, runners, split manifests and trained checkpoints.
- Implement finite PCM history and sample-index requests; check valid,
  overwritten, future and wraparound cases before using a model.
- Connect a Python request client, then integrate encoder, fixed retrieval and
  adaptive selection. Evaluate whether longer segments add value over context-only.

Compare feature/context-only, fixed refinement, uncertainty-only and deadline-aware
policies under comparable byte budgets and controlled link conditions. Report
timely event quality, boundary error, transmitted bytes, latency, deadline misses
and MCU memory/runtime. Define the reporting-deadline anchor before experiments.
Minimum target: a Nano/laptop USB prototype; paced replay is the microphone fallback.

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

## Starting points

- [Compression at the Edge for Energy- and Bandwidth-Efficient Industrial Audio Analysis](https://doi.org/10.1016/j.procs.2025.07.180)
- [Communication-Efficient Multi-Modal Edge Inference via Uncertainty-Aware Distributed Learning](https://arxiv.org/abs/2601.14942)
- [Wireless Evidence Acquisition for Multimodal Inference Constrained by Temporal Windows](https://arxiv.org/abs/2609.31428)

We build on feature compression and selective evidence acquisition, focusing on
temporal audio segments, finite MCU history and on-time boundary estimates.
