# Focused checks

Run all checks from the repository root with the project conda environment
active. They use synthetic inputs and need no downloads or checkpoints.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

- `test_scoring.py`: the 191-target 50 ms grid for a 10 s clip, half-open label
  overlap, and false/missed frame accounting.
- `test_pretrained_sed.py`: the 2023 model's 64 ms timestamp grid, exclusion of
  padded audio, overlapping class exports, threshold edges, and features scaled
  to [0, 1] as in its training code.
- `test_baseline.py` (earlier custom model): boundary labels, loader agreement on
  a synthetic PCM16 recording, finite outputs and gradients, no future-window
  leakage, state reset across recordings, waveform/spectrogram equivalence.

These are correctness checks, not accuracy results. When implementing the device
path, add checks for ring wraparound, overwritten and not-yet-recorded requests,
timestamp alignment and request/response parsing, using known sample values.
