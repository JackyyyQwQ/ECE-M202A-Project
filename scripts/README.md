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
time. Features follow the 2023 training code ([0, 1] per-clip scaling; see
`normalized_mel`). See the [comparison report](../docs/dcase2019_model_comparison.md). This runner does
not reproduce the original custom CNN+GRU experiments or implement online lookahead.


`fetch_dcase_pilot.py` retrieves 64 fixed, seed-2026 recordings and annotations
from the public ZIP via byte ranges, validating member CRCs and recording hashes.
`python -m scripts.evaluate_pretrained` evaluates the frozen official model on
that manifest using the 50 ms frame alignment, raw/smoothed controls and
matching rules described in the [comparison report](../docs/dcase2019_model_comparison.md).


For the entire corpus, use `fetch_dcase_pilot.py --all` (resumable archive download,
complete MD5 plus member CRC checks) and `python -m scripts.evaluate_pretrained --all`.
This uses a separate full manifest/output folder and preserves the pilot.
`report_pretrained.py` creates a Markdown/PDF report and two figures; it requires
`requirements-report.txt` and does not change the model or thresholds.
