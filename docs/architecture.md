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

The MCU retains bounded PCM history, emits timestamped features and returns
segments before overwrite. The laptop maintains causal state and chooses one
action per candidate; combined LOOKAHEAD/REFINE actions are outside the first version.

## Actions and training

- **REPORT:** submit the causal result without extra evidence.
- **LOOKAHEAD:** wait a declared horizon, then use a BiGRU over a bounded feature
  window to predict the original target interval. Later frames must actually
  arrive; full-clip backward context is forbidden for online evaluation.
- **REFINE:** fetch a historical PCM interval and correct the initial boundary.
  Post-fetch support scores are not free pre-request selection signals.

Train offline; deploy only the frontend/CNN to the MCU. Reproduce the causal
baseline first, then freeze its encoder for path comparisons. Joint fine-tuning
is a later option.

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

Present: custom causal CNN/GRU and refiner/context-only code. Audio refinement
has not beaten context-only overall. Separate official 2019/2023 CNN+BiGRUs run
offline with their own frontends and full future context, not our 32D features.
On 692 real public evaluation recordings, raw frame F1 is 70.87% (2023, with
feature scaling matched to its training code) versus 61.90% (2019); event macro F1
is 40.52% versus 28.37%. The 2019 model's filtered 28.97% matches its official 29.0%.
The 16 kHz 2023 model is the reference detector. This is an offline reference, not the deployed causal path or online latency.
See [baseline comparison](dcase2019_model_comparison.md) and
[original feasibility](feasibility.md).
Bounded-context adaptation, firmware, receiver and controller remain unimplemented.

Compare causal, bounded BiGRU, context-only and PCM paths under matched time/byte
budgets and capacity controls. Offline oracle selection is headroom analysis only.
Then test a simple controller using pre-action information; retain a fixed path
if it dominates. Measure timely quality, boundary error, bytes, misses and MCU costs.
Minimum hardware target: Nano + laptop over USB, with paced replay as microphone
fallback.
