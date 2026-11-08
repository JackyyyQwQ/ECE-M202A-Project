# Bounded-lookahead sweep (development set)

November 8, 2026. **Waiting helps, with sharply diminishing returns.** With the
2023 reference detector and causal hysteresis, waiting 0.5 s recovers 57% of the
event-F1 gap between reporting immediately and seeing the whole clip; 2 s
recovers 75% (micro) or 60% (class-macro). These are development-set results
with full-CNN features, i.e. lookahead with no compression loss.

## Setup

- **Data:** 916 of 1168 real DCASE2019 validation clips
  (re-downloaded from YouTube with `scripts/fetch_validation.py`; the rest are
  unavailable), 3,371 reference events. This is development data, used for tuning.
- **Detector:** official DCASE 2023 CNN+BiGRU (`legacy_epoch=176-step=20886.ckpt`), unchanged.
- **Online simulation** (`scripts/lookahead_sweep.py`): each 64 ms output frame is
  predicted from audio from the clip start up to L seconds after that frame, and
  only that frame is kept; features are normalized per window. "Full clip" is the
  offline reference. Added reporting delay is L plus one 64 ms frame; computation
  and transport are excluded.
- **Decision rules (all causal; frame t uses frames <= t only):** threshold 0.5;
  hysteresis (start at >= 0.5, end below 0.3); mean of the current and two previous
  frames, then 0.5. No centered median filter, which would add hidden lookahead.
- **Metrics:** frame F1 on the 50 ms grid; sed_eval event F1 with a 200 ms onset
  collar and an offset collar of max(200 ms, 20% of event length), reported
  class-macro (DCASE convention) and micro (all classes pooled; steadier here).

## Results

Hysteresis, the best rule at every lookahead:

| Lookahead | Frame F1 | Event F1 (micro) | Gap recovered (micro) | Event F1 (macro) | Gap recovered (macro) | Onset F1 (macro) | Offset F1 (macro) | Predicted events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 s | 0.636 | 0.229 | 0% | 0.144 | 0% | 0.229 | 0.352 | 7,056 |
| 0.5 s | 0.679 | 0.339 | 57% | 0.240 | 37% | 0.349 | 0.401 | 4,576 |
| 1 s | 0.688 | 0.356 | 66% | 0.266 | 47% | 0.376 | 0.424 | 4,080 |
| 2 s | 0.700 | 0.374 | 75% | 0.301 | 60% | 0.416 | 0.447 | 3,667 |
| full clip | 0.709 | 0.422 | 100% | 0.405 | 100% | 0.513 | 0.541 | 2,954 |

Event F1 (micro) by decision rule:

| Decision rule | 0 s | 0.5 s | 1 s | 2 s | Full clip |
| --- | ---: | ---: | ---: | ---: | ---: |
| threshold 0.5 | 0.187 | 0.315 | 0.330 | 0.346 | 0.412 |
| hysteresis 0.5/0.3 | 0.229 | 0.339 | 0.356 | 0.374 | 0.422 |
| causal mean (3 frames) | 0.169 | 0.335 | 0.352 | 0.372 | 0.417 |

Full table (all rules, onset/offset micro and macro): [`results/lookahead_validation_summary.json`](../results/lookahead_validation_summary.json).

## Findings

1. **Data check passed.** Full-clip event macro F1 with hysteresis is 0.405,
   matching the official 40.7% collar F1 on this set; frame F1 0.709 matches the
   public evaluation set. The re-downloaded audio is aligned with the labels.
2. **The first 0.5 s matters most.** It recovers 57% of the micro gap; 0.5 to 2 s
   adds 18 points more. Short waits are cheap and effective, so the deadline value
   will decide much of the controller's room.
3. **Waiting fixes boundaries more than frames.** Frame F1 rises only from 0.636
   to 0.700, while event F1 more than doubles. With no wait, predictions fragment
   into 7,056 events (about twice the 3,371 reference events);
   at 2 s the count is 3,667.
4. **Onsets and offsets both gain.** At 2 s, onset macro F1 rises 0.229 to 0.416
   and offset 0.352 to 0.447.
5. **Hysteresis is the best causal rule** at every lookahead and is the default
   for later experiments. Its thresholds (0.5/0.3) are untuned; tune them here.
6. **A gap remains at 2 s** (macro 0.301 vs 0.405). It comes from context more than
   2 s ahead. REFINE with the full CNN recovers compact-encoder loss, not this
   future context.

## Limits

- Full-CNN features: the compact Nano encoder will add its own loss; rerun with it.
- Development set only; the public evaluation set is reserved for frozen
  configurations. Re-downloaded clips may differ from the organizers' copies.
- Per-window normalization is one online choice; offline and online inputs
  therefore differ slightly even at large L.
- Reproduce: `python scripts/lookahead_sweep.py` (about 30 s of single-core
  compute per clip; six workers took about 1.5 h on a Mac). `--rescore` reuses
  the saved scores to try other decision rules.
