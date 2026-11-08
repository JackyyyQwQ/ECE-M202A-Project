# Focused checks

`test_baseline.py` checks half-open boundary labels, partial/overlapping events,
agreement between both loaders on a synthetic PCM16 recording, finite outputs
and gradients, absence of future-window leakage, state reset across recordings,
and equivalence of the waveform and cached-spectrogram paths.

From the repository root, with the project conda environment activated:

```bash
python -m unittest discover -s tests -p 'test_baseline.py' -v
```

These are correctness checks, not real-data training or F1 reproduction. They
use randomly initialized networks and temporary synthetic files. No download is
needed. No firmware test suite exists yet.

When implementing the device path, check exact sample retrieval across ring
wraparound, overwritten/future requests, timestamp and label alignment, and
request/response parsing. Use known synthetic sample values before real audio.
For model changes, verify causality and compare against context-only on fixed
recording splits. Preserve missed/false events and retrieval costs in evaluation.


With `requirements-pretrained.txt` installed, `test_pretrained_sed.py` checks the
64 ms timestamp grid, exclusion of padded audio, overlapping class exports and
threshold boundaries. It requires no checkpoints or downloads. Run all checks with:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```


`test_evaluate_pretrained.py` checks the original 191-target time grid, half-open
label overlap, one/two-sample archive duration rounding, false/missed frame accounting,
and one-to-one event matching without duplicate or wrong-class credit.
