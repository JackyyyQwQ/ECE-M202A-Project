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
| Bounded-lookahead BiGRU, online receiver and controller | Planned; compare paths before selection |

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
accuracy. Evaluating detection quality still requires the original training
artifacts and real audio.

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
