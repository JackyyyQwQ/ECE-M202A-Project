"""REFINE simulation: fetch raw PCM chunks around selected frames (offline, validation set).

The Nano normally sends compact student-CNN features (2 kB/s). Each clip may also
fetch PCM up to a budget (extra kB/s on top of the features). Selected frames are
expanded to chunks of +-`window` s (shrunk to fit the remaining budget, min +-0.128 s); the laptop runs the reference CNN on each
fetched chunk alone (zero padding at chunk edges, as it would online), replaces the
features of every frame in the chunk, and reruns the frozen BiGRU.

Selection policies, each spending the same per-clip budget:
  uncertainty  frames whose most uncertain class is closest to p = 0.5 first (online-feasible)
  random       random frames, mean of several seeds (control)
  oracle       frames where student and teacher outputs differ most first (NOT online-feasible;
               upper bound on what a better trigger could achieve)

    python scripts/refine_sim.py --run ds_w025

Still optimistic: chunk features use the full-clip [0, 1] normalisation; no deadline,
buffer expiry, lookahead or request overhead. Event F1 differences below ~0.01 are noise.
Writes output/distill/<run>/refine/{summary.json, refine_curve.png}.
"""

import argparse
import json
import time

import numpy as np
import torch

from distill_cnn import CACHE, OUT, annotations, cnn_input, head, load_teacher
from lookahead_sweep import POSTPROCESSING, score

FRAME_S = 0.064
PCM_BYTES_PER_S = 32000              # 16 kHz, 16-bit mono
FEATURE_BYTES_PER_S = 128 / FRAME_S  # 128-D int8 features every 64 ms = 2 kB/s


def split(array, frames):
    return np.split(array, np.cumsum(frames)[:-1])


def runs(mask):
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def select(order, n, k, budget, min_half=2):
    """Greedy: add frames in `order`, each as a chunk of +-k frames (shrunk to the remaining
    budget, never below +-min_half), while the fetched frames fit the budget."""
    mask = np.zeros(n, bool)
    for t in order:
        remaining = budget - mask.sum()
        half = min(k, (remaining - 1) // 2)
        if half < min_half:
            if remaining < 2 * min_half + 1:
                break
            continue
        lo, hi = max(0, t - half), min(n, t + half + 1)
        extra = (~mask[lo:hi]).sum()
        if extra and extra <= remaining:
            mask[lo:hi] = True
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--budgets", default="1,2,4,8", help="extra uplink for PCM, kB/s (PCM is 32 kB/s)")
    parser.add_argument("--windows", default="0.25,0.5,1", help="chunk half-length around each chosen frame, s")
    parser.add_argument("--random-seeds", type=int, default=3)
    parser.add_argument("--rule", default="hysteresis 0.5/0.3", choices=list(POSTPROCESSING))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    run = OUT / args.run
    data = np.load(run / "validation_features.npz")
    names = [str(n) for n in data["names"]]
    frames = data["frames"]
    student_f = split(data["student_feats"].astype(np.float32), frames)
    student_s = split(data["student_scores"], frames)
    teacher_s = split(data["teacher_scores"], frames)
    cache = np.load(CACHE / "validation.npz")
    assert [str(n) for n in cache["names"]] == names
    mel_all, mel_frames = cache["mel"], cache["mel_frames"]   # read each npz array once
    mels = [mel_all[i, :, :mel_frames[i]] for i in range(len(names))]
    usable = dict(zip(names, cache["usable"].tolist()))
    durations = dict(zip(names, cache["durations"].tolist()))
    total_s = sum(durations[n] for n in names)
    events = annotations()
    decide = POSTPROCESSING[args.rule]
    model, _ = load_teacher()

    def refined_scores(i, mask):
        if not mask.any():
            return student_s[i]
        feats = torch.from_numpy(student_f[i].copy())
        for a, b in runs(mask):  # reference CNN on the fetched chunk only
            chunk = torch.from_numpy(mels[i][:, 4 * a:4 * b]).float()[None]
            feats[a:b] = model.cnn(cnn_input(chunk)).squeeze(-1).permute(0, 2, 1)[0, :b - a]
        return head(model, feats[None])[0].numpy()

    def evaluate(label, masks, **info):
        with torch.inference_mode():
            outputs = {n: refined_scores(i, masks[i]) for i, n in enumerate(names)}
        r = score(names, outputs, events, durations, usable, decide)
        fetched = sum(m.sum() for m in masks) * FRAME_S
        row = {"policy": label, **info, "pcm_fraction": fetched / total_s,
               "uplink_kB_per_s": (FEATURE_BYTES_PER_S + PCM_BYTES_PER_S * fetched / total_s) / 1000,
               "event_micro_f1": r["event_micro_f1"], "event_macro_f1": r["event_macro_f1"],
               "frame_f1": r["frame_micro"]["f1"], "predicted_events": r["predicted_events"]}
        print(f"{label:<38} PCM {100 * row['pcm_fraction']:5.1f}%  uplink {row['uplink_kB_per_s']:5.1f} kB/s  "
              f"event uF1 {row['event_micro_f1']:.4f}  event F1 {row['event_macro_f1']:.4f}", flush=True)
        return row

    start = time.time()
    print(f"{len(names)} validation recordings; post-processing: {args.rule}")
    rows = [evaluate("student only", [np.zeros(f, bool) for f in frames], policy_kind="endpoint"),
            evaluate("all PCM (whole clip = teacher)", [np.ones(f, bool) for f in frames], policy_kind="endpoint")]
    uncertainty = [np.abs(s - 0.5).min(axis=1) for s in student_s]          # small = uncertain
    disagreement = [np.abs(s - t).max(axis=1) for s, t in zip(student_s, teacher_s)]
    for window in [float(x) for x in args.windows.split(",")]:
        k = int(round(window / FRAME_S))
        for budget in [float(x) for x in args.budgets.split(",")]:
            limits = [int(budget * 1000 / PCM_BYTES_PER_S * f) for f in frames]
            common = dict(window_s=window, budget_kB_per_s=budget)
            masks = [select(np.argsort(u, kind="stable"), len(u), k, b) for u, b in zip(uncertainty, limits)]
            rows.append(evaluate(f"uncertainty +{budget:g} kB/s, ±{window:g}s", masks, policy_kind="uncertainty", **common))
            masks = [select(np.argsort(-d, kind="stable"), len(d), k, b) for d, b in zip(disagreement, limits)]
            rows.append(evaluate(f"  oracle    +{budget:g} kB/s, ±{window:g}s", masks, policy_kind="oracle", **common))
            for seed in range(args.random_seeds):
                rng = np.random.default_rng(args.seed + seed)
                masks = [select(rng.permutation(len(u)), len(u), k, b) for u, b in zip(uncertainty, limits)]
                rows.append(evaluate(f"  random#{seed}   +{budget:g} kB/s, ±{window:g}s", masks,
                                     policy_kind="random", seed=seed, **common))

    out = run / "refine"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({
        "run": args.run, "recordings": len(names), "postprocessing": args.rule, "seconds": time.time() - start,
        "assumptions": "Offline, full-clip context; reference CNN run on each fetched chunk alone "
                       "(zero padding at chunk edges) but with full-clip [0,1] normalisation; oracle uses "
                       "teacher outputs and is not online-feasible; no deadline, buffer expiry, lookahead "
                       "or request overhead.",
        "rows": rows}, indent=2, default=float) + "\n")
    plot(rows, out / "refine_curve.png", args)
    print(f"Saved {out / 'summary.json'} and refine_curve.png")


def plot(rows, path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    windows = sorted({r["window_s"] for r in rows if "window_s" in r})
    styles = {"uncertainty": ("#2a78d6", "-", "o"), "random": ("#6b6a63", ":", "x"), "oracle": ("#eb6834", "--", "^")}
    fig, axes = plt.subplots(1, len(windows), figsize=(4.2 * len(windows), 4.4), dpi=200, sharey=True)
    axes = np.atleast_1d(axes)
    low, high = rows[0], rows[1]
    for ax, w in zip(axes, windows):
        for kind, (color, line, marker) in styles.items():
            pts = {}
            for r in rows:
                if r.get("window_s") == w and r.get("policy_kind") == kind:
                    pts.setdefault(r["budget_kB_per_s"], []).append((r["uplink_kB_per_s"], r["event_micro_f1"]))
            xy = sorted((np.mean([p[0] for p in v]), np.mean([p[1] for p in v])) for v in pts.values())
            xy = [(low["uplink_kB_per_s"], low["event_micro_f1"])] + xy
            ax.plot(*zip(*xy), line, marker=marker, color=color, lw=2 if kind == "uncertainty" else 1.4, ms=6,
                    label={"uncertainty": "uncertainty (online)", "random": "random (mean)",
                           "oracle": "oracle (not online)"}[kind])
        ax.axhline(high["event_micro_f1"], color="#2b2b28", lw=1, ls=(0, (4, 3)))
        ax.text(2.2, high["event_micro_f1"] + 0.001, "all PCM / teacher", fontsize=8, color="#2b2b28")
        ax.set_title(f"chunk ±{w:g} s", loc="left", fontsize=11)
        ax.set_xlabel("Average uplink (kB/s)")
        ax.grid(color="#e4e3dd"); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(f"Event F1 (micro), {args.rule}")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"REFINE with fetched PCM chunks (run {args.run})", x=0.01, ha="left")
    fig.text(0.01, 0.01, "Preliminary, offline simulation · 916 validation clips · 2 kB/s features + fetched PCM "
             "(32 kB/s) · full-clip normalisation · differences < ~0.01 F1 are noise", fontsize=7, color="#6b6a63")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path)


if __name__ == "__main__":
    main()
