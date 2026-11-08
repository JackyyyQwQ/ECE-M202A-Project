# Architecture

P13 · Junchi Chen & Wei Zhang · Revised November 7, 2026

**Goal:** measure when waiting for future features or retrieving buffered audio
improves on-time event boundaries, then test whether adaptive selection beats
fixed choices. Design rationale: [proposal](proposal.md).

## Intended runtime

```text
Nano + Zephyr                              Laptop + Python
mic / paced replay -> PCM ring buffer
                   -> log-mel -> compact CNN --features--> 2023 BiGRU + heads
                                                              |
                                                          controller
                             +--------------------------------+-------------+
                             |                                |             |
                          REPORT                          LOOKAHEAD       REFINE
                     window ends now               wait L s, rerun     request PCM,
                                                   bounded BiGRU     full CNN on it
                             +--------------------------------+-------------+
                                                              |
                                                     event + onset/offset
```

**Detector.** The official DCASE 2023 CNN+BiGRU (16 kHz, 128-mel, 64 ms frames)
is the reference; it runs without training. Its full CNN (~84 M MAC/s, estimated)
is too large for the Nano's Cortex-M4F, so the Nano will run a compact CNN trained
to reproduce the reference CNN's 128-D frame features (~2 KB/s at int8). The BiGRU
and output heads are reused unchanged on the laptop.

**Actions.** REPORT uses a window ending at the current frame (no future context).
LOOKAHEAD waits L seconds for later features and reruns the BiGRU on the bounded
window; only features that have actually arrived may be used. REFINE requests a
PCM interval before it is overwritten and replaces the compact features for that
interval with the full CNN's features. The first controller version chooses one
action per candidate, using only pre-action information.

## Execution and timing

| Mode | Device | Laptop | Purpose |
|---|---|---|---|
| Offline | Python, recorded WAV | Python paths | Quality vs lookahead and evidence value |
| Simulation (planned) | Zephyr native_sim, paced replay | Python receiver | Buffer and protocol integration |
| Hardware (planned) | Nano mic or paced replay | Same receiver over USB | Real runtime, memory, transfer |

Native_sim does not measure Cortex-M execution time, radio performance or energy.

Audio is 16 kHz mono PCM16. Planned messages: `FEATURE(sequence, sample interval,
values)`, `AUDIO_REQUEST(id, start_sample, count)`, `AUDIO_RESPONSE(id, status,
PCM)`, with half-open sample intervals and status distinguishing available,
overwritten and not-yet-recorded samples. Waiting, feature/PCM transfer, queueing
and inference all consume reporting time; waiting also shortens the remaining
buffer lifetime of the audio REFINE would need. Buffer expiry and the reporting
deadline are separate constraints; define deadline anchors and clock mapping first.

## Evidence and next steps

On the 692-recording DCASE2019 public evaluation set with full-clip context, the
2023 model reaches 70.87% frame F1 and 40.52% event macro F1 (2019 cross-check:
61.90% / 28.37%, reproducing its published 29.0% filtered score). These are
offline upper references; see [detector evaluation](dcase2019_model_comparison.md).
An earlier custom CNN + causal GRU with a 200 ms refiner did not beat a
context-only control ([feasibility](feasibility.md)).

Next: (1) lookahead sweep (L = 0, 0.5, 1, 2 s) on development data; (2) Nano PCM
buffer and request protocol; (3) compact encoder and REFINE; (4) controller under
matched byte and time budgets, keeping a fixed path if it dominates. Tune only on
development data; score the public set once per frozen configuration. Minimum
hardware target: Nano + laptop over USB, with paced replay as microphone fallback.
