# Controller simulation (policy A)

November 11, 2026. **Preliminary, offline replay of the development (validation) set,
916 clips.** Rule-based controller, no training. Script:
[`scripts/controller_sim.py`](../scripts/controller_sim.py); results:
[`results/controller/ds_w025_summary.json`](../results/controller/ds_w025_summary.json).

## Summary

- **Early report** (report a frame as soon as every class is confident) cuts the mean
  reporting delay by 40-45% (1.00 -> 0.60 s at D = 1 s) with about 78-80% of frames
  reported early. At the same deadline it costs 0.001-0.006 event micro F1 and
  0.01-0.02 macro F1; at the same *mean* delay it is more accurate than waiting a
  fixed time.
- **REFINE** gives a small, consistent gain: event F1 rises with the budget in all six
  series, by about +0.01 micro and +0.01-0.015 macro at +8 kB/s (2 -> about 9.5 kB/s
  uplink). This is at the noise threshold; a bootstrap test is pending.
- **Policy A** (early report + waiting + REFINE) at D = 2 s, +8 kB/s: mean delay
  1.01 s, event F1 0.356 micro / 0.250 macro, vs 0.338 / 0.226 for fixed waiting
  with the same mean delay (D = 1 s).
- Online results stay well below the whole-clip student (0.390 / 0.339): the main
  loss is missing future context, which REFINE cannot recover.

## Setup

**System simulated.** The Nano sends int8 student features (width 0.25 DS-CNN, 128-D
every 64 ms, 2 kB/s; see [distillation and REFINE](distillation_refine.md)). Every 64 ms
tick the laptop reruns the frozen official BiGRU on the features that have arrived and
decides for its unreported frames. Frames are reported in order and never revised;
events come from causal hysteresis (0.5 / 0.3) on the reported frames.

**Timing.** The per-frame student needs 20 future mel frames plus half an FFT window,
so frame t's feature arrives about 0.38 s after the frame ends: no frame can be
reported earlier. REFINE costs an assumed round trip R = 0.32 s (request, PCM over the
link, laptop compute; not measured). D and R are whole 64 ms ticks.

**Rule.** u = the distance from 0.5 of the class closest to 0.5 (small = uncertain).

1. u >= `m_rep` -> REPORT now (early report).
2. Age >= deadline D -> REPORT.
3. At age D - R, once per frame: if u < `m_ref`, the PCM chunk t +- 0.25 s is
   recorded and still in the 2 s ring buffer, and the token bucket can pay for the
   newly fetched part -> REFINE. The laptop keeps fetched PCM and reruns the reference
   CNN on each contiguous fetched run; the result is used from age D.
4. Otherwise LOOKAHEAD: wait one more tick.

The token bucket refills at the budget b (kB/s on top of the 2 kB/s features) across
the whole stream of clips; depth 10 s of budget.

**Policies.** Two switches give four policies: fixed L (every frame waits until D;
baseline), early report, fixed L + refine, and policy A (both).

| Parameter | Value | Role |
|---|---|---|
| D | 1, 1.5, 2 s | condition |
| b | 0, 2, 4, 8 kB/s | condition |
| `m_rep` | 0.3 (every class <= 0.2 or >= 0.8), or infinity | early report |
| `m_ref` | 0.25 | refine trigger |
| REFINE chunk | +-0.25 s (9 frames, 18 KB) | from the offline REFINE study |
| R | 0.32 s | assumption |

**Sanity check.** Waiting for every feature of the clip reproduces the whole-clip
student exactly (0.3896 micro / 0.3386 macro).

## Results

Event F1 micro / macro and mean delay; +8 kB/s rows shown (all budgets in the JSON).

| Policy | D = 1 s | D = 1.5 s | D = 2 s |
|---|---|---|---|
| fixed L | 0.338 / 0.226, 1.00 s | 0.342 / 0.239, 1.41 s | 0.351 / 0.253, 1.85 s |
| early report | 0.335 / 0.217, 0.60 s | 0.341 / 0.228, 0.78 s | 0.344 / 0.235, 1.01 s |
| fixed L + refine, +8 kB/s | 0.346 / 0.235, 1.00 s | 0.350 / 0.252, 1.41 s | 0.361 / 0.265, 1.85 s |
| policy A, +8 kB/s | 0.345 / 0.232, 0.61 s | 0.348 / 0.241, 0.79 s | 0.356 / 0.250, 1.01 s |

Reference points: whole-clip student 0.390 / 0.339; whole-clip teacher 0.422 / 0.405.

- **Lookahead:** fixed L from D = 1 to 2 s adds +0.013 micro and +0.027 macro.
- **Early report:** 95th-percentile delay stays at D (1.02 / 1.47 / 1.98 s); the gain
  is in the mean. Compared at matched mean delay it dominates fixed waiting
  (D = 1.5 s, 0.78 s: 0.341 / 0.228 vs fixed L D = 1 s, 1.00 s: 0.338 / 0.226).
- **REFINE:** the budget is almost fully spent (uplink close to 2 + b). Frames that
  trigger a refine: about 0.7 / 2.7 / 7.5% at 2 / 4 / 8 kB/s; each fetches 9 frames of PCM.
- **Noise:** a single event micro F1 on 916 clips is about +-0.017 (95%); paired
  differences between policies on the same clips are smaller (estimated +-0.005, not
  yet measured).

## Limitations

- Offline replay; whole-clip [0, 1] normalisation; float student features (int8
  per-frame features differ by less than noise).
- R fixed at 0.32 s; no packet loss or jitter; Nano compute and feature transfer time
  not included in the delays.
- Thresholds were set by hand and evaluated on the development set; the public set is
  reserved for one final run.
- Early reports are never revised; their flip rate is not yet measured.

## Reproduce

```bash
python scripts/distill_cnn.py cache                    # once, if not done
python scripts/distill_cnn.py eval --run ds_w025       # student features (committed weights)
python scripts/controller_sim.py --run ds_w025 --limit 20   # quick check
python scripts/controller_sim.py --run ds_w025              # all 916 clips, ~10 min
```

## Next

Bootstrap confidence intervals for policy differences; shorter deadlines
(D = 0.45 / 0.6 / 0.75 s); sweep `m_rep` and measure the early-report flip rate; plot
mean delay vs event F1; replace R with a measured BLE round trip.
