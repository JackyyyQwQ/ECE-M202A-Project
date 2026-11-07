# Architecture

P13 · Junchi Chen & Wei Zhang · Revised November 7, 2026

**Goal:** compare timely audio-inference paths, then determine whether adaptive
selection improves on fixed choices. Full design: [proposal](proposal.md).

## Intended runtime

```text
Nano + Zephyr                           Laptop + Python
mic / paced replay -> PCM history
                  -> CNN --features--> causal GRU -> initial prediction
                                                  |
                                              controller
                          +-----------------------+--------------------+
                          |                       |                    |
                       REPORT                 LOOKAHEAD              REFINE
                          |                wait for features     request PCM
                          |                bounded-window BiGRU      |
                          |                       |             audio + context
                          |                       |                refiner
                          +-----------------------+--------------------+
                                                  |
                                               report
```

The MCU continuously retains bounded PCM history, emits timestamped compact
features and returns requested segments before overwrite. The laptop maintains
causal state and chooses one action per candidate initially; repeated or combined
LOOKAHEAD/REFINE actions are outside the first version.

## Actions and training

- **REPORT:** submit the causal result without extra evidence.
- **LOOKAHEAD:** wait a declared horizon, then use a BiGRU over a bounded feature
  window to predict the original target interval. Later frames must actually
  arrive; full-clip backward context is forbidden for online evaluation.
- **REFINE:** fetch a historical PCM interval and correct the initial boundary.
  Post-fetch support scores are not free pre-request selection signals.

Train models offline; deploy only the compact frontend/CNN to the MCU. Improve
and reproduce the causal baseline first. Initially freeze the same encoder for
path comparisons; shared/jointly fine-tuned heads are a later option, not implemented.

## Execution and timing

| Mode | Device | Laptop | Purpose |
|---|---|---|---|
| Offline | Python encoder/WAV | Python paths | Compare quality and evidence value |
| Simulation (planned) | Zephyr native_sim/replay | Python receiver | Buffer/protocol integration |
| Hardware (planned) | Nano/Zephyr mic or replay | Same receiver/USB | Measure actual resource costs |

Docker supplies build tools; it does not run on the Nano. Native_sim does not
measure Cortex-M execution time, radio performance or energy.

Current input: 16 kHz mono PCM16; 500 ms/8,000-sample windows; 50 ms/800-sample
hop; ten labels for the latest 50 ms. First output: 0.5 s, first 450 ms unscored.
Encoder: 32D (128D also tested). Existing refiner: exactly 200 ms, +/-100 ms correction.
Future horizons and PCM lengths remain experiment choices.

Planned messages: FEATURE(sequence, sample interval, values), AUDIO_REQUEST(id,
start_sample, count), AUDIO_RESPONSE(id, status, PCM). Use half-open sample
intervals; distinguish available, overwritten and not-yet-recorded samples.
Define deadline anchors and clock mapping before evaluation. Waiting, feature/PCM
transfer, queueing and inference all consume time. Buffer extraction expiry is
separate from reporting deadline; waiting also reduces remaining audio history.

## Evidence and next steps

Present: custom causal CNN/GRU, refiner/context-only and correctness checks.
Audio refinement has not beaten context-only overall. A separate official
pretrained CNN+BiGRU now runs offline on Mac: 30.0 ms warm median compute for a
padded 10-second example, no accuracy evaluation. It uses its own CNN/mel, not our
32D features. See [trial](pretrained_baseline.md) and [feasibility](feasibility.md).
Bounded-context adaptation, firmware, receiver and controller remain unimplemented.

First compare causal, bounded BiGRU, context-only and PCM-refined results under
matched time/byte budgets and capacity controls. Use offline oracle selection only
as headroom analysis. Then test a simple controller using pre-action information;
retain a fixed path if it dominates. Measure timely event quality, boundary error,
bytes, misses and MCU costs. Minimum hardware target is Nano + laptop over USB,
with paced replay as microphone fallback.
