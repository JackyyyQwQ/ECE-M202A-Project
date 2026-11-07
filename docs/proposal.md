# Deadline-Aware Collaborative Audio Inference

P13 · Junchi Chen and Wei Zhang · ECE M202A / CS M213A, Fall 2026
Revised November 7, 2026. This is a proposed design, not a completed system.

## Research question and setting

In continuous sound-event detection, when should a receiver report a causal
prediction, wait for bounded future features and run a bidirectional GRU, or
retrieve original audio from an MCU's finite buffer? We will first measure the
quality, latency and communication trade-offs of these paths, then investigate
whether a deadline-aware controller improves on fixed choices under changing
communication conditions and reporting budgets.

The device is an Arduino Nano 33 BLE Sense running Zephyr. A lightweight CNN
encodes microphone audio into compact features; a laptop performs temporal
detection, optional refinement and action selection. USB is the initial transport.

## Motivation and starting points

Compact features reduce continuous communication but may discard boundary detail.
Waiting can provide future temporal context, while retrieving PCM can provide
additional acoustic detail. Both consume reporting time; waiting also ages the
MCU's retained history. Their complementarity is a hypothesis to test.

We build on audio compression, uncertainty-guided feedback, and evidence-value
scheduling. Dynamic inference and lookahead are established ideas: SPINN studies
runtime inference scheduling, while sequential SED work studies future-context
latency. Our proposed focus is the choice between waiting for future features and
retrieving expiring historical audio for event boundaries. We do not claim that
GRUs, adaptive inference, or uncertainty feedback are new. See
[related work](references.md) for sources and distinctions.

## Proposed approach

The MCU continuously maintains PCM history and emits timestamped CNN features.
A laptop-side unidirectional GRU produces initial activity predictions and event
boundaries. A controller will choose one action per candidate in the first version:

| Action | Evidence | Cost |
|---|---|---|
| REPORT | Current causal prediction | Base processing |
| LOOKAHEAD | Features through a bounded future horizon, processed by a windowed BiGRU | Waiting, feature transport and extra computation |
| REFINE | Requested PCM segment plus available context | Request, queueing, audio transfer and refinement |

The BiGRU must predict the same target interval as the causal path, using only
features actually received by its bounded horizon. Full-recording bidirectional
inference is not an online result. The refiner remains a separate optional laptop
module. Existing support scores are post-retrieval and cannot select requests for
free. Model sizes, lookahead lengths and segment sizes will be chosen on development
data; the current 32D and 200 ms settings are starting points.

Training is offline; only the compact encoder is deployed to the MCU. We will
improve the causal baseline first, then initially hold the encoder fixed across
path comparisons. Joint fine-tuning is optional subsequent work and must preserve
causal-path quality. We do not assume pretrained GRUs accept arbitrary CNN features.

## Evaluation before controller design

1. Recover the original runners, checkpoints and recording splits; reproduce and
   diagnose the causal baseline.
2. Compare causal GRU, bounded-lookahead BiGRU, context-only correction, and
   causal GRU plus PCM refinement. Measure quality, decision time and bytes.
3. Check whether the best affordable path changes across samples and conditions.
   An offline oracle may estimate selection headroom but is not a deployable policy.
4. If useful headroom exists, implement a small rule-based controller using only
   initial predictions and current resource estimates. A learned policy is optional,
   not required. If one fixed path dominates, report that result and simplify.

Use identical splits, class ordering and controlled link traces. Compare at
matched reporting budgets and explicit byte budgets; count base features, extra
features, PCM, metadata and unsuccessful attempts as applicable. Compare refinement
against a similar-duration lookahead and context-only control. Control model
capacity where possible, or add an ablation so extra parameters are not mistaken
for future-context gains.

Primary outcome: timely event F1 with declared class/boundary tolerances. Also
report boundary MAE, false/missed events, bytes, latency, deadline misses, and
measured MCU memory/runtime. Define onset/offset deadline anchors and clock mapping
before evaluation; ground-truth times cannot drive online actions. Waiting,
queueing, computation and transport all count. Buffer extraction expiry is distinct
from the final report deadline. Stress-test link delay/bandwidth and buffer lifetime.

## Minimum deliverable, scope and fallback

Deliver a Nano/laptop prototype with compact features, finite PCM history, a causal
detector, bounded-lookahead and refinement comparisons, and a simple resource-aware
selector. The controller need not outperform fixed paths to make the evaluation
informative. If microphone integration fails, replay recordings at their original
rate on the physical MCU. If extra evidence has no useful advantage, preserve the
working baseline and report where added latency/communication fails to pay off.

Wireless transport, custom neural-network novelty, and multi-round combinations
of LOOKAHEAD and REFINE are outside the required scope. One stretch goal is a
learned action selector, conditional on working paths and demonstrated headroom.

## Current evidence and responsibilities

Implemented offline: loaders, causal CNN/GRU, fixed 200 ms audio refinement and
context-only control. Preliminary audio refinement did not beat context-only
overall. Seven synthetic correctness checks passed; these do not establish real
accuracy. A separate official pretrained CNN+BiGRU was subsequently smoke-tested offline
without training; see [trial](pretrained_baseline.md). It is a candidate baseline
with its own frontend, not an integrated bounded-lookahead path. Its project-data
accuracy is unknown. No online controller or project hardware loop is implemented.
See the preserved [feasibility evidence](feasibility.md).

- **Wei:** recover experiment artifacts; improve causal detection; compare
  bounded-lookahead and refinement, including time alignment and fair controls.
- **Junchi:** implement Zephyr PCM history, replay/capture and USB request handling;
  measure resource costs and support the receiver/controller integration.
- **Together:** lock deadline definitions, interfaces and evaluation; decide whether
  adaptive selection is justified. For Checkpoint 3, show existing evidence and
  this staged plan, with new measurements labeled separately.
