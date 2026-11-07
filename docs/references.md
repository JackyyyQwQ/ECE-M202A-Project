# Related work and how we build on it

Updated November 7, 2026. The adaptations below are our proposed work, not results
established by the cited papers. They do not constitute an exhaustive novelty review.

## Core related work

### 1. Audio compression at the edge

**Compression at the Edge for Energy- and Bandwidth-Efficient Industrial Audio
Analysis** (2025), Procedia Computer Science 265, 260–267.
[Paper / DOI](https://doi.org/10.1016/j.procs.2025.07.180)

- **Idea:** evaluate compressed audio through industrial anomaly detection and
  sound-event segmentation, alongside bandwidth and hardware energy costs.
- **Build on:** measure downstream event quality and transmitted bytes for our
  compact features, then test whether selective raw audio recovers useful detail.
- **Difference:** our learned encoder and feedback policy require their own
  validation; the paper's compression and energy results do not transfer directly.
- **Experiment:** fixed detector, varying feature size/precision, with and without
  refinement. Include all feature and retrieval bytes.

### 2. Uncertainty-guided feedback

**Communication-Efficient Multi-Modal Edge Inference via Uncertainty-Aware
Distributed Learning** — Hang Zhao et al. (2026), arXiv preprint.
[Paper](https://arxiv.org/abs/2601.14942)

- **Idea:** devices transmit representations; the server estimates per-modality
  uncertainty and selectively requests retransmission/additional features.
  Evaluation includes RGB–depth scene classification.
- **Build on:** an uncertainty-only fixed-segment retrieval baseline.
- **Difference:** choose temporal PCM segments from continuous audio, including
  their location and length. Uncertainty alone does not establish retrieval value.
- **Experiment:** compare uncertainty-only requests to a policy that also uses
  expected improvement, buffer availability and reporting time, at comparable cost.

### 3. Evidence value under a time window

**Wireless Evidence Acquisition for Multimodal Inference Constrained by Temporal
Windows** — Alessandro Compagnoni et al. (2026), arXiv preprint.
[Paper](https://arxiv.org/abs/2609.31428)

- **Idea:** MIRA prioritizes conditional evidence gain while considering channel
  conditions and remaining data volume within a finite acquisition window.
- **Build on:** rank useful evidence against the time/bytes needed to obtain it.
- **Difference:** its LMMSE-based information metric cannot simply be applied to
  our neural boundary refiner. We must estimate audio's incremental value and
  handle overwritten or not-yet-recorded segments in finite MCU history.
- **Experiment:** remove the value, channel/delay and buffer constraints one at
  a time. Evaluate quality at the deadline, not just delivered data volume.

### 4. Separate boundary extent from event confidence

**Sound Event Bounding Boxes** — Janek Ebbers, François G. Germain, Gordon Wichern
and Jonathan Le Roux (2024), Interspeech.
[Paper](https://arxiv.org/abs/2406.04212) ·
[Author code](https://github.com/merlresearch/sebbs)

- **Idea:** represent an event's class, onset, offset and presence confidence
  separately; changing confidence thresholds should not arbitrarily move boundaries.
- **Build on:** distinguish confidence that an event exists from uncertainty in
  its boundaries before spending bytes on a request.
- **Difference:** the change-detection postprocessor uses context on both sides
  of a boundary. An online adaptation must account for future-context waiting.
- **Experiment:** compare current frame-threshold boundaries with an offline
  diagnostic or bounded-lookahead alternative, charging any lookahead delay.

## Supporting methods

### 5. CNN + recurrent sound-event detection

**Convolutional Recurrent Neural Networks for Polyphonic Sound Event Detection** —
Emre Çakır et al. (2017).
[Paper](https://arxiv.org/abs/1702.06286)

- **Idea:** CNNs extract local time/frequency features; recurrent layers model
  temporal context for overlapping sound events.
- **Build on:** retain a small CNN and a causal GRU as our detection backbone.
  The CNN/GRU architecture is a foundation, not our novelty claim.
- **Experiment:** hold the backbone fixed when comparing request policies.

### 6. Confidence calibration

**On Calibration of Modern Neural Networks** — Chuan Guo, Geoff Pleiss, Yu Sun
and Kilian Q. Weinberger (2017), ICML, PMLR 70, 1321–1330.
[Paper](https://proceedings.mlr.press/v70/guo17a.html)

- **Idea:** neural-network confidence may not reflect correctness; post-hoc
  calibration, including temperature scaling, can improve probability estimates.
- **Build on:** check confidence reliability on a separate development split.
  Consider calibration before using scores as a request signal.
- **Difference:** results do not automatically transfer to our class-weighted,
  multi-label sigmoid outputs. Classification calibration is not boundary-value
  estimation and does not guarantee extra audio will help.
- **Experiment:** compare request yield before/after calibration; fit gates and
  calibration on development data only, never on the held-out test set.

## Proposed synthesis

For candidate segments, first check whether samples can be acquired before
overwrite and whether retrieval/refinement can finish on time. Then compare
expected improvement over **context-only** against acquisition cost, using only
pre-request information. Ground truth may supervise offline value estimation but
must not select online request locations. Compare context-only, fixed retrieval,
uncertainty-only and deadline-aware policies on the same recordings/link traces.

Our current audio refiner has not beaten context-only overall. First establish
whether 200/400 ms evidence adds value; only then evaluate adaptive selection.
See [architecture](architecture.md) and [feasibility evidence](feasibility.md).
