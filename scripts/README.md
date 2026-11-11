# Experiment entry points

The original preprocessing, training and evaluation runners have not been supplied.
Recover these from the team's existing experiment files before claiming reproduction.
Add small entry points here as they become available; document exact commands,
configurations, data splits, checkpoints and output locations.

Current data inspection: `python -m data.audio_dataset`, from the repository root
after installing dependencies and supplying the files listed in `data/README.md`.
Model imports now use `models.activity` / `models.boundary_refinement`.
`from models import CausalActivityModel` is retained by the package initializer.

## Official pretrained detectors (no training)

Run from the repository root with `conda activate "$PWD/.conda"`.

| Script | Purpose |
|---|---|
| `fetch_public_eval.py` | Download the 692-recording DCASE2019 public evaluation set (MD5 and SHA-256 checked) |
| `evaluate_public.py --model 2023\|2019` | Score a frozen checkpoint on that set; writes `output/public_eval/<model>/` |
| `verify_public.py --model 2023\|2019` | Recount raw TP/FP/FN from cached scores; needs no weights |
| `lookahead_sweep.py` | Simulate bounded lookahead (L = 0, 0.5, 1, 2 s) with the 2023 detector; `--limit N` for a quick check |
| `fetch_validation.py` | Download the real 1,168-clip validation set from YouTube (resumable; development data) |
| `dcase2019_model.py` | 2019 checkpoint loader and 44.1 kHz frontend (imported) |
| `pretrained_sed.py` | 2023 checkpoint loader/frontend; single-recording inference |
| `scoring.py` | Shared 50 ms frame targets and P/R/F1 (imported) |

Setup and commands: [comparison report](../docs/dcase2019_model_comparison.md#evidence-and-reproduction).

`pretrained_sed.py` runs the official 2023 CNN+BiGRU offline, without
training. It writes timestamped frame scores, events and measured CPU computation
time. Features follow the 2023 training code ([0, 1] per-clip scaling; see
`normalized_mel`). See the [comparison report](../docs/dcase2019_model_comparison.md). This runner does
not reproduce the original custom CNN+GRU experiments or implement online lookahead.

## Training audio

`fetch_dcase_pilot.py` downloads the DCASE2019 synthetic corpus (2,045 recordings,
with `--all`; default: a fixed 64-recording subset) into `data/raw/dcase2019/` via
byte ranges, checking the archive MD5 and member CRCs. It is kept as the audio
source for training the compact Nano encoder; its earlier 2023-model scores used
the old feature scaling and were removed.

`scoring.py` holds the shared 50 ms frame-target and precision/recall/F1 helpers.

## Compact encoder and REFINE (offline prototypes)

| Script | Purpose |
|---|---|
| `distill_cnn.py cache\|train\|eval` | Distil a DS-CNN student from the 2023 reference CNN (BiGRU frozen); `eval` scores it on validation and saves features. Committed students: `models/distill/<run>/` |
| `refine_sim.py --run <run>` | Fetch PCM chunks around uncertain frames under an uplink budget, recompute them with the reference CNN; uncertainty vs random vs oracle |
| `export_tflite.py --run <run>` | int8 TFLite export (per-frame 40 mel-frame window), F1 of float/int8 on validation, C header for `firmware/tflm_probe` |
| `controller_sim.py --run <run>` | Online controller simulation: early report, waiting and REFINE under a deadline and token-bucket budget |

Results and commands: [distillation, int8 and REFINE](../docs/distillation_refine.md),
[controller](../docs/controller.md).
