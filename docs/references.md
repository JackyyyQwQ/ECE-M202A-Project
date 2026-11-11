# Related work and how we build on it

Updated November 11, 2026. The adaptations below are our proposed work, not results
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

### 7. Adaptive device/cloud inference

**SPINN: Synergistic Progressive Inference of Neural Networks over Device and
Cloud** — Stefanos Laskaridis et al. (2020), MobiCom.
[Paper](https://arxiv.org/abs/2008.06402)

- **Idea:** combine progressive inference and device/cloud splitting, adapting
  execution to runtime conditions and latency/accuracy requirements.
- **Build on:** profile alternative inference paths before choosing a simple
  runtime policy. Adaptive path selection itself is not our novelty.
- **Difference:** we compare acquiring future temporal context with retrieving
  expiring historical PCM for event boundaries, rather than only choosing where
  to execute a network or where to exit.
- **Experiment:** compare the selector against fixed affordable paths across link
  conditions, including the cost of the selection decision.

### 8. Future context in streaming sound-event detection

**A Sequential Audio Spectrogram Transformer for Real-Time Sound Event Detection**
— Takezo Ohta, Yoshiaki Bando, Keisuke Imoto and Masaki Onishi (2024), EUSIPCO.
[Paper](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000101.pdf)

- **Idea:** sequential chunk processing with memory tokens supports streaming SED;
  experiments examine the trade-off between future context and latency.
- **Build on:** explicitly bound and charge future-context waiting, evaluating
  the original target interval after its permitted evidence arrives.
- **Difference:** our proposed temporal head is a bounded-window BiGRU, and we
  compare waiting against additional raw-audio acquisition. Neither lookahead nor
  its accuracy/latency trade-off is a new contribution by itself.
- **Experiment:** sweep finite horizons and compare with PCM refinement at similar
  reporting times; control model capacity and prohibit full-recording future access.

## Added November 11, 2026

### 9. Server-driven re-sending of uncertain regions (REFINE)

**Server-Driven Video Streaming for Deep Learning Inference (DDS)** — Kuntai Du et al.
(2020), SIGCOMM. [Paper](https://kuntaidu.github.io/assets/doc/DDS.pdf)

- **Idea:** the camera streams low-quality video; the server finds low-confidence
  regions and asks the camera to re-encode only those at high quality.
- **Build on:** REFINE follows this two-stream design: cheap features by default, raw
  PCM only around uncertain frames, with early reporting of confident results.
- **Difference:** we request time segments around event boundaries; the PCM expires in
  a 2 s MCU buffer, and REFINE competes with waiting for future features.

### 10. Adaptive lookahead per frame

**Lookahead When It Matters: Adaptive Non-causal Transformers for Streaming Neural
Transducers** — Grant Strimel et al. (2023), ICML.
[Paper](https://proceedings.mlr.press/v202/strimel23a.html)

- **Idea:** a streaming ASR model decides per frame whether to wait for future frames.
- **Build on:** our controller waits only for uncertain frames and reports the rest early.
- **Difference:** rule-based on a frozen detector; waiting also shortens the buffer
  lifetime of the PCM a REFINE would need.

### 11. Stopping feature transmission when it no longer pays

**Progressive Feature Transmission for Split Inference at the Wireless Edge** — Qiao Lan,
Qunsong Zeng, Petar Popovski, Deniz Gündüz and Kaibin Huang (2021), arXiv.
[Paper](https://arxiv.org/abs/2112.07244)

- **Idea:** the server requests more features until the expected uncertainty reduction no
  longer justifies the communication cost; the optimal policy is a threshold rule.
- **Build on:** threshold triggers (`m_ref`) under an explicit byte budget.

### 12. Compact encoder and early exit

- **FitNets: Hints for Thin Deep Nets** — Romero et al. (2015), ICLR.
  [Paper](https://arxiv.org/abs/1412.6550). Student fits a teacher's intermediate
  features; our DS-CNN fits the reference CNN's 128-D features.
- **MobileNets** — Howard et al. (2017). [Paper](https://arxiv.org/abs/1704.04861);
  **Hello Edge: Keyword Spotting on Microcontrollers** — Zhang et al. (2017).
  [Paper](https://arxiv.org/abs/1711.07128). Depthwise-separable CNNs and the width
  multiplier; DS-CNN as the standard MCU audio model.
- **BranchyNet** — Teerapittayanon et al. (2016). [Paper](https://arxiv.org/abs/1709.01686).
  Confidence-based early exit; our early report applies the idea in time.

## Proposed synthesis

Status (November 11): the reference detector is the official DCASE 2023 CNN+BiGRU,
used without training; the Nano runs a distilled DS-CNN that reproduces its CNN
features; the BiGRU stays unchanged on the laptop. Offline prototypes of the encoder,
REFINE and a rule-based controller are done ([encoder](distillation_refine.md),
[controller](controller.md)).

The controller compares REPORT (report a frame as soon as it is confident), LOOKAHEAD
(wait for more features, up to a deadline) and REFINE (fetch the buffered PCM around an
uncertain frame and recompute its features with the reference CNN), under a deadline,
a 2 s MCU buffer and a link budget. Baselines: fixed waiting, uncertainty-only REFINE,
random selection and an offline oracle, all with the same budgets. Tune only on the
development set; score the public set once.

Our proposed contribution is an empirical study of **waiting for future features
versus retrieving expiring historical audio** for timely event boundaries on an
MCU/laptop system. This combination is a research hypothesis, not a verified claim of
being first. If one fixed path dominates, report that result and simplify the system.
See [proposal](proposal.md), [architecture](architecture.md) and
[feasibility evidence](feasibility.md).
