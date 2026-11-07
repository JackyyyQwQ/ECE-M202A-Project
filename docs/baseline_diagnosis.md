# Baseline diagnosis — November 7, 2026

## Scope

The teammate's preprocessing/training/evaluation runners, actual split manifests,
trained checkpoints and audio are not available locally. We cannot reproduce
real F1, inspect a trained model's normalization, or identify the main cause of
the reported poor precision yet. The checks below are synthetic correctness
checks; the model architecture, loader logic and original metrics were not changed.

## Confirmed environment problem

The default interpreter was Anaconda base: PyTorch 2.0.1 with NumPy 2.4.6.
`torch.from_numpy` failed with `RuntimeError: Numpy is not available`; pandas
import failed with a binary dtype-size mismatch. This prevents local execution
but does not explain experiments previously run in the teammate's environment.

Use the independent project conda environment (`environment.yml`, local `.conda/`)
instead of changing base or other course environments. An initial `.venv/` was
used for an intermediate check and is ignored; it is not the recommended setup.
Both local environment directories are excluded from Git.

## Correctness checks

`tests/test_baseline.py` covers seven cases:

1. An event [0.50, 0.55) labels only the corresponding frame; touching endpoints
   do not count as overlap.
2. Partial overlaps, simultaneous classes, and events before scored coverage.
3. Both loaders agree on all 191 windows/labels of a synthetic ten-second PCM16
   recording. WAV-to-float scaling and the sample offsets agree with known values.
4. 32D and 128D models produce finite logits and gradients, including silence.
5. Changing future windows does not change earlier predictions.
6. A previous recording/call does not affect predictions for a new recording/call.
7. Cached spectrogram and waveform paths give matching predictions.

These checks do not validate the missing real files, training losses, learned
weights, class calibration, event matching or measured accuracy.

**Result:** all seven checks passed in the independent conda environment;
`python -m pip check` found no broken requirements. Tested versions: Python
3.11.18, torch 2.15.0, NumPy 1.26.4, pandas 2.3.3, SoundFile 0.14.0. The
intermediate environment (torch 2.0.1 / NumPy 1.26.4) also passed the seven checks.

## Leading hypotheses, not confirmed root causes

**Class weighting plus a fixed 0.5 threshold.** The report describes positive
weights equal to negative/positive frame counts, high recall, and low precision.
For a single binary label under ideal weighted BCE, the optimum weighted score is
`q = w*p / (1-p+w*p)`. Thresholding q at 0.5 corresponds to p >= 1/(w+1), not
p >= 0.5. Thus weighting can substantially shift the operating point toward
positives. This is a mathematical diagnostic, not a calibration guarantee for
the trained network. Recover logits and sweep class thresholds on development
data, retaining the test set untouched. Compare precision-recall and event metrics.

**Limited fitting data or incomplete optimization.** The baseline report uses 128
training recordings. Check learning curves and per-class positives; attempt to
fit a tiny real training subset before increasing data or network size.

**Unknown training integration.** Definitions initialize normalization buffers to
zero mean/unit scale, but the missing runner/checkpoint may overwrite them.
Inspect saved means/stds, finite inputs, BCEWithLogitsLoss vs sigmoid usage,
class order, recording splits, batch sequence boundaries and checkpoint selection.
Do not infer from default buffer values that normalization was absent in training.

**Representation and event conversion.** The encoder pools a linear-frequency
log spectrum to 32x16 and ultimately 4x2 before its linear bottleneck. It is not
a log-mel frontend. Whether this loses necessary boundary detail is untested.
Likewise, thresholding can fragment or merge events. Hold the model fixed while
checking thresholds/postprocessing; charge any lookahead to reporting latency.

## Next required inputs

- Training/evaluation scripts and selected recording filenames/class order.
- Checkpoint containing encoder/GRU weights and normalization buffers.
- Configuration with loss weights, sampling, seed and selection criterion.
- Real validation logits/targets/timestamps, or enough data to regenerate them.

Then reproduce the reported baseline, inspect per-class precision-recall, and
test thresholds before making architecture changes. No improved F1 is claimed.
