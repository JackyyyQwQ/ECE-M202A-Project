# Experiment entry points

The original preprocessing, training and evaluation runners have not been supplied.
Recover these from the team's existing experiment files before claiming reproduction.
Add small entry points here as they become available; document exact commands,
configurations, data splits, checkpoints and output locations.

Current data inspection: `python -m data.audio_dataset`, from the repository root
after installing dependencies and supplying the files listed in `data/README.md`.
Model imports now use `models.activity` / `models.boundary_refinement`.
`from models import CausalActivityModel` is retained by the package initializer.


`pretrained_sed.py` runs a separate official pretrained CNN+BiGRU offline, without
training. It writes timestamped frame scores, events and measured CPU computation
time. See [setup and limitations](../docs/pretrained_baseline.md). This runner does
not reproduce the original custom CNN+GRU experiments or implement online lookahead.


`fetch_dcase_pilot.py` retrieves 64 fixed, seed-2026 recordings and annotations
from the public ZIP via byte ranges, validating member CRCs and recording hashes.
`python -m scripts.evaluate_pretrained` evaluates the frozen official model on
that manifest. See [evaluation](../docs/pretrained_evaluation.md) for the 50 ms
alignment, raw/smoothed controls, matching rules and comparison limitations.
