# Deadline-Aware Collaborative Audio Inference

**P13 · UCLA ECE M202A / CS M213A · Fall 2026** · Junchi Chen · Wei Zhang

[Project website](https://jackyyyqwq.github.io/ECE-M202A-Project/) ·
[Proposal](docs/proposal.md) · [Architecture](docs/architecture.md) ·
[Detector evaluation](docs/dcase2019_model_comparison.md) ·
[Related work](docs/references.md)

## Overview

A low-power acoustic sensor should report not only *what* happened but *when* it
started and ended, within a deadline. An Arduino Nano 33 BLE Sense running Zephyr
cannot run an accurate detector, and streaming raw audio is costly, so the Nano
keeps a short PCM history and sends compact features to a laptop, which detects
sound events and their onset/offset times.

For an uncertain boundary the laptop can **report** now, **wait** for more future
features (accurate detectors use future context), or **retrieve** the buffered raw
audio from the Nano before it is overwritten. **Research question:** when is waiting
or retrieving worth its time and bytes, under reporting deadlines, a finite MCU
buffer and a communication budget? Adaptive selection is a hypothesis to test.

## Status

| Component | Status |
|---|---|
| Reference detector: official DCASE 2023 CNN+BiGRU (no training) | Done, offline: 70.87% frame F1 on 692 real recordings |
| Cross-check: official DCASE 2019 CNN+BiGRU | Done: reproduces its published 29.0% event F1 |
| Bounded-lookahead sweep (wait 0 to 2 s) | Done on 916 validation clips ([results](docs/lookahead_sweep.md)) |
| Nano PCM ring buffer | Done in native_sim: 13-case self-test on a synthetic signal ([firmware](firmware/README.md)) |
| Request protocol, laptop link, real-audio replay | Next |
| Compact MCU encoder, PCM refinement, controller | Planned |
| Earlier custom CNN + causal GRU and 200 ms refiner | Preliminary feasibility only ([report](docs/feasibility.md)) |

## Results so far

Frozen official checkpoints on all 692 real recordings of the DCASE2019 public
evaluation set, threshold 0.5, no training or tuning, full-clip (offline) context:

| Model | Frame precision | Frame recall | Frame F1 | Event macro F1 |
|---|---:|---:|---:|---:|
| 2019 CNN+BiGRU (44.1 kHz) | 69.00% | 56.13% | 61.90% | 28.37% |
| **2023 CNN+BiGRU (16 kHz)** | 78.06% | 64.90% | **70.87%** | **40.52%** |

The 2019 model with its median filter gives 28.97% event F1 (published: 29.0%).
The 2023 model is the reference detector: it is stronger, and its 16 kHz input
matches the Nano capture rate. Its checkpoint expects features scaled to [0, 1] as
in its 2023 training code; the newer upstream code uses [-1, 1], which had
suppressed detections (39.20% frame F1) until fixed. These are offline upper
references, not online or MCU results. Details, provenance and limitations:
[detector evaluation](docs/dcase2019_model_comparison.md); scores:
[`results/public_eval_summary.json`](results/public_eval_summary.json).

**Lookahead (development set, 916 real validation clips).** Simulating the 2023
detector online, with causal hysteresis, event F1 (micro) is 0.229 when reporting
immediately, 0.339 after waiting 0.5 s, 0.374 after 2 s and 0.422 with the whole
clip: the first 0.5 s recovers 57% of the gap, with sharply diminishing returns.
Waiting mainly repairs fragmented events and their boundaries; frame F1 changes
little. Full-clip results match the official score on this set, confirming the
re-downloaded audio. Details: [lookahead sweep](docs/lookahead_sweep.md).

The earlier custom pipeline (CNN + causal GRU, 200 ms refiner) is kept as a
feasibility study: audio refinement did not beat a context-only control
(372.4 vs 373.6 ms boundary MAE); see [feasibility](docs/feasibility.md).

## Setup

One conda environment for all offline work (macOS arm64 tested), from the
repository root:

```bash
conda env create --prefix ./.conda --file environment.yml   # once
conda activate "$PWD/.conda"                                 # every session
```

It installs [`requirements-pretrained.txt`](requirements-pretrained.txt) (torch,
torchaudio, librosa, sed_eval). Data, checkpoints, upstream code and outputs stay
in Git-ignored `data/raw/`, `checkpoints/`, `cache/` and `output/`; the
[detector evaluation](docs/dcase2019_model_comparison.md#evidence-and-reproduction)
lists how to obtain each.

## Reproduce

```bash
python scripts/fetch_public_eval.py                  # 692 WAVs, checksums verified
python scripts/evaluate_public.py --model 2023       # or --model 2019
python scripts/verify_public.py --model 2023         # independent recount
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/pretrained_sed.py recording.wav --output output/one_file   # one clip
```

## Repository map

```text
scripts/      Detector loaders, public evaluation, downloads (see scripts/README.md)
models/       Earlier custom CNN/GRU and boundary refiner
data/         Data sources, download notes and custom-model loaders
tests/        Feature range, timestamp export and custom-model checks
results/      Committed score summaries and small figures
docs/         Proposal, one-page architecture, evaluations, related work
firmware/     Zephyr app, toolchain image and launcher (ring buffer done)
receiver/     Laptop runtime and controller (planned)
configs/      Experiment settings (added with working runners)
index.html    Project website (plain HTML, no build step)
```

## Hardware and simulation

1. **Offline Python**: recorded audio through the detectors. Current work.
2. **Zephyr native_sim + Python receiver**: the ring buffer runs and passes its
   self-test in native_sim (`bash firmware/run sim`); replay, protocol and the
   Python link are next.
3. **Nano + Zephyr + Python receiver over USB**: microphone or paced replay. Planned.

The toolchain comes from the
[course Zephyr repository](https://github.com/UCLA-ECEM202A/software/tree/83a3ac6ccf585b2b647d3d6c654b616de8c29cb8/zephyr);
see [firmware/README.md](firmware/README.md). Native_sim does not measure Nano speed,
radio performance or energy.

## Next steps

1. **Lookahead sweep.** Done on the development set: waiting pays, mostly in the
   first 0.5 s ([results](docs/lookahead_sweep.md)). Next: tune hysteresis there and
   define the reporting deadline.
2. **Device path.** Ring buffer done (valid, overwritten, not-yet-recorded,
   wraparound and slow-transfer cases pass in native_sim). Next: native_sim to
   Python link, request protocol and real-audio replay.
3. **Compact encoder and refinement.** A small CNN on the Nano trained to reproduce
   the reference CNN's features (the BiGRU is reused unchanged); retrieved PCM is
   re-encoded by the full CNN on the laptop.
4. **Controller.** Compare report / wait / retrieve under matched byte and time
   budgets; keep a fixed policy if it dominates.

Evaluation rules: tune thresholds, lookahead and controller only on development
data; score the public evaluation set once per frozen configuration. Count waiting,
queueing, transfer and inference in latency, and define deadline anchors first.

## Development

Keep code small and readable; keep [docs/architecture.md](docs/architecture.md) to
one page. Do not commit audio, caches, environments or checkpoints. Preview the
website with `python3 -m http.server 8000 --bind 127.0.0.1` and open
`http://127.0.0.1:8000/`.
