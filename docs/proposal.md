# Deadline-Aware Collaborative Audio Inference

P13 · Junchi Chen and Wei Zhang · ECE M202A / CS M213A, Fall 2026
Revised November 7, 2026. This is a proposed design, not a completed system.

## Research question and setting

A low-power acoustic sensor must report sound events with their onset and offset
times within a deadline. Accurate detectors are non-causal: the official DCASE
2023 CNN+BiGRU, our reference detector, reads the whole clip. Online, a receiver
must therefore choose, for each uncertain boundary, whether to **report** with no
future context, **wait** a bounded time for later features, or **retrieve** the
original audio from an MCU's finite buffer before it is overwritten. We will
measure the quality, latency and communication trade-offs of these paths for this
detector, then test whether a deadline-aware controller improves on fixed choices.

The device is an Arduino Nano 33 BLE Sense running Zephyr, connected to a laptop
over USB initially. The research contribution is the online adaptation and action
selection around a fixed detector, not a new neural network.

## Motivation and starting points

The Nano cannot run the full detector (its CNN needs an estimated ~84 M MAC/s),
and streaming raw 16 kHz PCM costs 32 KB/s. Sending compact features saves
bandwidth but may lose boundary detail; waiting supplies future context; retrieving
PCM restores acoustic detail. Both consume reporting time, and waiting also ages
the audio the MCU still holds. Their complementarity is a hypothesis to test.

We build on audio compression, uncertainty-guided feedback, evidence-value
scheduling, split computing and lookahead in sequential detection. We do not claim
that GRUs, adaptive inference or uncertainty feedback are new; our focus is the
choice between waiting for future features and retrieving expiring historical
audio for event boundaries. See [related work](references.md).

## Proposed approach

**Detector (no training).** Official DCASE 2023 CNN+BiGRU: 16 kHz, 128-mel input,
64 ms frames, ten DESED classes ([evaluation](dcase2019_model_comparison.md)).

**Split.** The Nano keeps a PCM ring buffer, computes log-mel features and runs a
compact CNN trained to reproduce the reference CNN's 128-D frame features
(~2 KB/s at int8). Only this encoder is trained; the BiGRU and output heads are
reused unchanged on the laptop. The full reference CNN is the accuracy upper bound
for the compact encoder.

**Actions.** A controller chooses one action per candidate in the first version:

| Action | Evidence | Cost |
|---|---|---|
| REPORT | BiGRU on a window ending at the current frame | Base processing |
| LOOKAHEAD | BiGRU on a window extended by L seconds of received features | Waiting and extra computation |
| REFINE | Requested PCM, re-encoded by the full CNN on the laptop | Request, queueing, transfer and computation |

LOOKAHEAD may use only features that have actually arrived; full-clip inference is
an offline reference, not an online result. If REPORT (L = 0) proves too weak, a
unidirectional GRU distilled from the BiGRU is an optional alternative.

## Evaluation plan

1. **Lookahead sweep.** Frame/event F1 and boundary error for L = 0, 0.5, 1, 2 s
   versus full context, with the reference CNN. This measures what waiting can buy.
2. **Compact encoder.** Quality loss of the Nano encoder relative to the full CNN,
   and how much REFINE recovers per retrieved second of audio.
3. **Headroom.** Whether the best affordable action changes across boundaries and
   conditions. An offline oracle estimates headroom but is not a deployable policy.
4. **Controller.** If headroom exists, a small rule-based controller using only
   pre-action information; a learned policy is optional. If one fixed path
   dominates, report that and simplify.

Tune thresholds, lookahead and controller settings only on development data; score
the 692-recording public evaluation set once per frozen configuration. Compare at
matched reporting and byte budgets under controlled link traces; count features,
PCM, metadata and unsuccessful attempts. Compare REFINE against a similar-duration
LOOKAHEAD.

Primary outcome: timely event F1 with declared class and boundary tolerances. Also
report boundary error, false/missed events, bytes, latency, deadline misses and
measured MCU memory/runtime. Define onset/offset deadline anchors and clock mapping
before evaluation; ground-truth times cannot drive online actions. Buffer expiry is
distinct from the report deadline. Stress-test link delay/bandwidth and buffer size.

## Minimum deliverable, scope and fallback

A Nano/laptop prototype with compact features, finite PCM history, the reference
BiGRU with bounded lookahead, REFINE, and a simple resource-aware selector. The
controller need not beat fixed paths for the evaluation to be informative. If the
microphone path fails, replay recordings at their original rate on the Nano. If
the compact encoder cannot run in real time, report its measured cost and use paced
replay of precomputed features. Wireless transport, new network architectures and
multi-round LOOKAHEAD/REFINE combinations are out of scope; a learned selector is a
stretch goal.

## Current evidence and responsibilities

On the DCASE2019 public evaluation set with full-clip context, the 2023 detector
reaches 70.87% frame F1 and 40.52% event macro F1 without training; the 2019
detector reproduces its published score (28.97% vs 29.0% filtered event F1). An
earlier custom CNN + causal GRU with a 200 ms refiner did not beat a context-only
control ([feasibility](feasibility.md)). No online path, encoder, controller or
hardware loop is implemented yet.

- **Wei:** lookahead sweep, compact-encoder training and REFINE evaluation, with
  time alignment and fair controls.
- **Junchi:** Zephyr PCM ring buffer, replay/capture and USB request handling;
  measure MCU costs; receiver integration.
- **Together:** lock deadline definitions, interfaces and the development split;
  decide whether adaptive selection is justified. For Checkpoint 3, show the
  reference-detector evidence, the lookahead results and this staged plan.
