# Experiment entry points — pending

The original preprocessing, training and evaluation runners have not been supplied.
Recover these from the team's existing experiment files before claiming reproduction.
Add small entry points here as they become available; document exact commands,
configurations, data splits, checkpoints and output locations.

Current data inspection: `python -m data.audio_dataset`, from the repository root
after installing dependencies and supplying the files listed in `data/README.md`.
Model imports now use `models.activity` / `models.boundary_refinement`.
`from models import CausalActivityModel` is retained by the package initializer.
