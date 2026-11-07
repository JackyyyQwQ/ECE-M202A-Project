# Architecture

P13 · Junchi Chen & Wei Zhang · Updated November 7, 2026

**Goal:** improve sound-event onset/offset estimates through useful, timely audio
requests under a finite MCU buffer and limited communication budget.

## Intended runtime

```text
Device: Nano + Zephyr                 Laptop: Python
microphone / paced replay
  -> PCM ring buffer
  -> frontend + CNN encoder --features--> causal GRU -> event boundaries
          |                                            |
          |<----------- request(start, count) ----- request controller
          +-- buffered PCM / unavailable ----------> refiner -> report
```

**Device responsibilities:** continuously acquire audio, retain bounded history,
emit timestamped features, and copy requested samples before they are overwritten.
**Laptop responsibilities:** detect events, estimate request value, select segment
location/length, refine boundaries, and submit a result on time. The refiner is
part of the laptop pipeline. The offline training class currently includes both
the encoder and GRU; they are not yet separate deployed processes.

## Three modes

| Mode | Device side | Laptop side | Purpose |
|---|---|---|---|
| Offline | Python encoder, recorded WAV | Python detector/refiner | Model and audio-value experiments |
| Zephyr simulation (planned) | native_sim, paced replay | Python receiver | Buffer, protocol and integration checks |
| Hardware (planned) | Nano + Zephyr, mic/replay | Same receiver, USB | Actual runtime, memory and transport measurements |

Docker supplies the Linux compiler and native simulator. It does not run on the
Nano. Native simulator timing is not a Cortex-M performance measurement.

## Data and timing

16 kHz mono PCM16, float32 when loaded for training. Model input: 500 ms / 8,000
samples; hop: 50 ms / 800 samples; ten activity labels for the latest 50 ms.
First output is at 0.5 s; the first 450 ms is unscored. Current encoder features
are 32D (128D was also tested); the refiner supports exactly 200 ms of audio.

Planned messages: `FEATURE(sequence, start_sample, end_sample, values)`,
`AUDIO_REQUEST(id, start_sample, sample_count)`, and `AUDIO_RESPONSE(id, status,
samples)`. Status must distinguish available, overwritten, and not-yet-recorded
audio. Audio intervals are half-open. Transport and serialization remain undecided.

The controller must use only information available before requesting. Waiting for
future samples, queueing, transfer and refinement all consume reporting time.
Buffer extraction expiry and report deadlines are separate constraints. The
deadline anchor and clock alignment must be specified before timing experiments;
ground-truth timestamps are for evaluation only.

## Status and next steps

**Present:** offline loaders, CNN/GRU and refiner; preliminary results in
[feasibility.md](feasibility.md). Audio refinement has not beaten context-only
overall. No project firmware, transport or controller is implemented here.

**Next:** recover original runners/data/checkpoints; implement and check bounded
PCM history; connect a Python request client; integrate encoder, fixed retrieval,
then adaptive selection. Compare feature/context-only, fixed, uncertainty-only
and deadline-aware policies using timely event quality, boundary error, bytes,
latency, misses and measured MCU resources. Minimum target: Nano + laptop over
USB; paced replay is the fallback for microphone integration.
