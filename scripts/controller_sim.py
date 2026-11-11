"""Controller simulation (policy A): wait, report early when confident, REFINE as a last resort.

Offline, on the validation (development) set, with the distilled student's features.
Every 64 ms tick the laptop reruns the frozen BiGRU on the features that have arrived
and decides for the earliest unreported frame t (frames are reported in order):

  u = min over classes |p - 0.5|                    (small = uncertain)
  1. u >= m_report                       -> REPORT now (early exit)
  2. age >= deadline D                   -> REPORT now
  3. at age D - R (the last tick a REFINE can still meet D), for every waiting frame,
     not only the earliest: u < m_refine, PCM chunk [t-w, t+w] recorded and still in the
     2 s ring buffer, not fully fetched before, token bucket can pay
                                         -> REFINE: the laptop reruns the reference CNN on
                                            the contiguous PCM it now holds around t, then
                                            the BiGRU; the frame is REPORTed at age D
  4. otherwise                           -> LOOKAHEAD (wait one more frame)
When every feature of the clip has arrived there is nothing left to wait for, so steps
3/2 apply at once. R is the assumed REFINE round trip (request, PCM transfer, compute);
D and R are rounded to whole 64 ms ticks (R up).

Timing: the student needs 20 future mel frames plus half an FFT window, so feature s
arrives 6.75 frames after the start of frame s (ceil -> usable at tick s + 7, i.e.
>= 0.38 s after frame s ends). Uplink = 2 kB/s features + fetched PCM (32 kB/s);
the token bucket refills at the budget rate across the whole stream of clips.

    python scripts/controller_sim.py --run ds_w025 --limit 20   # quick check
    python scripts/controller_sim.py --run ds_w025

Sanity row: "student, whole clip" must equal the student's full-clip score (0.390 event
micro F1 for ds_w025); --sanity-refine adds "refine everything", which should approach
the teacher (0.422). Optimistic as before: whole-clip [0, 1]
normalisation, no packet loss, fixed R. Writes output/controller/<run>/summary.json.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from distill_cnn import CACHE, OUT, ROOT, annotations, cnn_input, load_teacher
from lookahead_sweep import POSTPROCESSING, score

FRAME_S = 0.064
AVAIL = 7                     # feature s usable at tick s + AVAIL
BUFFER_FRAMES = int(2.0 / FRAME_S)   # 2 s PCM ring buffer
PCM_BYTES_PER_FRAME = 2048    # 64 ms of 16-bit 16 kHz audio
FEATURE_KB_PER_S = 2.0
ROW_BLOCK = 8                 # BiGRU prefix rows recomputed per batch after a REFINE


def prefix_table(model, feats, ks):
    """Frozen BiGRU frame probabilities for each prefix length in ks: [len(ks), max(ks), classes]."""
    f = torch.from_numpy(feats)
    with torch.inference_mode():
        out, _ = model.rnn.rnn(pack_sequence([f[:k] for k in ks], enforce_sorted=False))
        out, _ = pad_packed_sequence(out, batch_first=True)
        return torch.sigmoid(model.dense(out)).numpy()


def chunk_features(model, mel, a, b):
    """Reference-CNN features of frames [a, b) computed from that PCM chunk alone."""
    with torch.inference_mode():
        chunk = torch.from_numpy(mel[:, 4 * a:4 * b]).float()[None]
        return model.cnn(cnn_input(chunk)).squeeze(-1).permute(0, 2, 1)[0, :b - a].numpy()


class Bucket:
    def __init__(self, rate_kbs, capacity_bytes):
        self.rate = rate_kbs * 1000 * FRAME_S          # bytes per frame
        self.capacity = capacity_bytes
        self.level, self.clock = 0.0, 0.0

    def advance(self, now):
        if now > self.clock:
            self.level = min(self.capacity, self.level + self.rate * (now - self.clock))
            self.clock = now


def simulate_clip(c, model, cfg, bucket, offset):
    """Policy A on one clip, one 64 ms tick at a time. Returns reported probs, delays (ticks), stats."""
    T, base = c["T"], c["table"]
    Dk, Rk, W = cfg["Dk"], cfg["Rk"], cfg["W"]
    feats = None                       # student features with applied REFINE results
    pending = []                       # (ready_tick, [(a, b, teacher features)])
    cache = {}                         # prefix length -> probabilities, valid for the current feats
    fetched = np.zeros(T, bool)
    tried = np.zeros(T, bool)
    ready = np.full(T, -1)
    reported = np.zeros((T, base.shape[2]), np.float32)
    delays = np.zeros(T)
    stats = {"refines": 0, "early": 0, "bytes": 0}

    def row(k):
        if feats is None:
            return base[k - 1]
        if k not in cache:
            ks = list(range(k, min(T, k + ROW_BLOCK - 1) + 1))
            table = prefix_table(model, feats, ks)
            for i, kk in enumerate(ks):
                cache[kk] = table[i]
        return cache[k]

    head, n = 0, 0
    while head < T:
        k = min(T, n - AVAIL + 1)
        if k <= head:
            n += 1
            continue
        done = k >= T                                   # every feature of the clip has arrived
        due = [item for item in pending if item[0] <= n]
        if due:                                         # REFINE results that have come back
            if feats is None:
                feats = c["student"].copy()
            for _, parts in due:
                for a, b, f in parts:
                    feats[a:b] = f
            pending = [item for item in pending if item[0] > n]
            cache.clear()
        P = row(k)

        # 1. REFINE decisions: each frame at its own last feasible tick (or at once when done)
        decide_t = n - 1 - (Dk - Rk)
        candidates = [t for t in range(max(head, decide_t), k)
                      if not tried[t] and (t == decide_t or done)]
        chosen = []
        for t in candidates:
            tried[t] = True
            u = float(np.abs(P[t].astype(np.float32) - 0.5).min())
            lo, hi = max(0, t - W), min(T, t + W + 1)
            cost = int((~fetched[lo:hi]).sum()) * PCM_BYTES_PER_FRAME
            bucket.advance(offset + min(n, T))
            recorded = hi <= n                          # chunk audio has been captured
            buffered = lo >= n - BUFFER_FRAMES          # chunk start not yet overwritten
            if u < cfg["m_report"] and u < cfg["m_refine"] and cost and recorded and buffered and bucket.level >= cost:
                bucket.level -= cost
                stats["bytes"] += cost
                stats["refines"] += 1
                fetched[lo:hi] = True
                ready[t] = n + Rk
                chosen.append(t)
        if chosen:   # the laptop keeps fetched PCM: recompute each contiguous fetched run once
            edges = np.diff(np.r_[0, fetched.astype(int), 0])
            runs = [(a, b) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
                    if any(a <= t < b for t in chosen)]
            pending.append((n + Rk, [(a, b, chunk_features(model, c["mel"], a, b)) for a, b in runs]))

        # 2. report frames in order
        while head < k:
            t, age = head, n - (head + 1)
            p = P[t].astype(np.float32)
            if ready[t] >= 0:
                if n < ready[t]:
                    break                               # waiting for its REFINE result
            elif float(np.abs(p - 0.5).min()) >= cfg["m_report"]:
                stats["early"] += age < Dk
            elif not (age >= Dk or (done and tried[t])):
                break                                   # LOOKAHEAD
            reported[t], delays[t] = p, max(age, 0)
            head += 1
        n += 1
    return reported, delays, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="ds_w025")
    parser.add_argument("--deadlines", default="1,1.5,2", help="D, seconds after the frame ends")
    parser.add_argument("--budgets", default="0,2,4,8", help="extra uplink for PCM, kB/s")
    parser.add_argument("--m-report", default="0.3,inf", help="early-report margins (inf = never early)")
    parser.add_argument("--m-refine", default="0.25", help="refine only if u is below this")
    parser.add_argument("--window", type=float, default=0.25, help="REFINE chunk half-length, s")
    parser.add_argument("--rtt", type=float, default=0.3, help="assumed REFINE round trip R, s")
    parser.add_argument("--bucket-s", type=float, default=10.0, help="token bucket depth, s of budget")
    parser.add_argument("--limit", type=int, help="first N recordings only (quick check)")
    parser.add_argument("--sanity-refine", action="store_true",
                        help="also run 'refine everything' (slow: one BiGRU rerun per frame)")
    args = parser.parse_args()

    data = np.load(OUT / args.run / "validation_features.npz")
    names = [str(n) for n in data["names"]]
    frames = data["frames"]
    split = lambda a: np.split(a, np.cumsum(frames)[:-1])
    student = split(data["student_feats"].astype(np.float32))
    student_scores = split(data["student_scores"])
    cache = dict(np.load(CACHE / "validation.npz"))
    usable = dict(zip(names, cache["usable"].tolist()))
    durations = dict(zip(names, cache["durations"].tolist()))
    if args.limit:
        names = names[:args.limit]
    events = annotations()
    model, _ = load_teacher()
    decide = POSTPROCESSING["hysteresis 0.5/0.3"]

    print(f"{len(names)} validation recordings: building BiGRU prefix tables ...", flush=True)
    start = time.time()
    clips = []
    for i, n in enumerate(names):
        f = student[i]
        T = len(f)
        table = prefix_table(model, f, list(range(1, T + 1))).astype(np.float16)
        if i == 0:   # full prefix must equal the whole-clip student scores
            diff = float(np.abs(table[T - 1].astype(np.float32) - student_scores[0]).max())
            assert diff < 2e-3, f"prefix table differs from the whole-clip BiGRU by {diff}"
        clips.append({"name": n, "T": T, "student": f, "table": table,
                      "mel": cache["mel"][i, :, :cache["mel_frames"][i]]})
    print(f"  done in {time.time() - start:.0f} s", flush=True)

    W = int(round(args.window / FRAME_S))
    total_s = sum(c["T"] for c in clips) * FRAME_S

    def run(label, D, budget, m_report, m_refine):
        cfg = {"Dk": int(round(D / FRAME_S)), "Rk": math.ceil(args.rtt / FRAME_S - 1e-9), "W": W,
               "m_report": m_report, "m_refine": m_refine}
        capacity = max(budget * 1000 * args.bucket_s, (2 * W + 1) * PCM_BYTES_PER_FRAME) if budget else 0
        bucket = Bucket(budget, capacity)
        outputs, delays, total = {}, [], {"refines": 0, "early": 0, "bytes": 0}
        offset = 0.0
        for c in clips:
            rep, d, s = simulate_clip(c, model, cfg, bucket, offset)
            offset += c["T"]
            outputs[c["name"]] = rep
            delays.append(d)
            for key in total:
                total[key] += s[key]
        r = score([c["name"] for c in clips], outputs, events, durations, usable, decide)
        d = np.concatenate(delays) * FRAME_S
        nframes = len(d)
        row = {"policy": label, "deadline_s": D, "budget_kB_per_s": budget, "m_report": m_report,
               "m_refine": m_refine, "event_micro_f1": r["event_micro_f1"], "event_macro_f1": r["event_macro_f1"],
               "frame_f1": r["frame_micro"]["f1"], "mean_delay_s": float(d.mean()),
               "p95_delay_s": float(np.percentile(d, 95)),
               "uplink_kB_per_s": FEATURE_KB_PER_S + total["bytes"] / total_s / 1000,
               "refined_frame_share": total["refines"] / nframes, "early_share": total["early"] / nframes}
        print(f"{label:<44} uF1 {row['event_micro_f1']:.4f}  F1 {row['event_macro_f1']:.4f}  "
              f"delay {row['mean_delay_s']:.2f}/{row['p95_delay_s']:.2f} s  uplink {row['uplink_kB_per_s']:4.1f} kB/s  "
              f"refined {100 * row['refined_frame_share']:4.1f}%  early {100 * row['early_share']:4.1f}%", flush=True)
        return row

    inf = float("inf")
    rows = [run("sanity: student, whole clip", 1e6, 0, inf, 0.0)]
    if args.sanity_refine:
        rows.append(run("sanity: refine everything (D long)", 1e6, 1e6, inf, 1.0))
    for D in [float(x) for x in args.deadlines.split(",")]:
        for m_report in [float(x) for x in args.m_report.split(",")]:
            for budget in [float(x) for x in args.budgets.split(",")]:
                for m_refine in ([0.0] if budget == 0 else [float(x) for x in args.m_refine.split(",")]):
                    kind = "fixed L" if m_report == inf and budget == 0 else (
                        "fixed L + refine" if m_report == inf else (
                            "early report" if budget == 0 else "policy A"))
                    label = f"D={D:g}s {kind:<16} m_rep={m_report:g} b={budget:g}"
                    if budget:
                        label += f" m_ref={m_refine:g}"
                    rows.append(run(label, D, budget, m_report, m_refine))

    out = ROOT / "output/controller" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({
        "run": args.run, "recordings": len(clips), "window_s": args.window, "rtt_s": args.rtt,
        "bucket_s": args.bucket_s, "feature_avail_frames": AVAIL, "postprocessing": "hysteresis 0.5/0.3",
        "assumptions": "Offline replay of the validation set; whole-clip [0,1] normalisation; fixed REFINE "
                       "round trip; no packet loss; frames reported in order and never revised.",
        "rows": rows}, indent=2, default=float) + "\n")
    print(f"Saved {out / 'summary.json'}")


if __name__ == "__main__":
    main()
