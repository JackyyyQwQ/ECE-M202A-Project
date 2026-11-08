"""Independently recount a public-evaluation run from its cached scores.

    python scripts/verify_public.py --model 2023    # or --model 2019

Rebuilds the 50 ms targets from the official annotations and audio lengths,
recounts frame TP/FP/FN from output/public_eval/<model>/scores.npz, and checks
them against metrics.json and the per-recording totals. Needs no model weights.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.scoring import activity_targets, classification_counts, target_grid  # noqa: E402

DATA = ROOT / "data/raw/dcase2019_public_eval"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("2019", "2023"), required=True)
    args = parser.parse_args()
    out = ROOT / "output/public_eval" / args.model
    metrics = json.loads((out / "metrics.json").read_text())
    cache = np.load(out / "scores.npz")
    step_s = 0.064 if args.model == "2023" else 8 * 511 / 44100
    median = "median7" if args.model == "2023" else "median5"

    annotations = defaultdict(list)
    with (DATA / "public.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            annotations[row["filename"]].append((row["event_label"], float(row["onset"]), float(row["offset"])))
    with (out / "per_recording.tsv").open() as handle:
        per_file = list(csv.DictReader(handle, delimiter="\t"))
    frames = {r["filename"]: int(r["frames"]) for r in per_file if r["mode"] == "raw"}

    names = [str(name) for name in cache["filenames"]]
    assert len(names) == 692 and len(set(names)) == 692
    counts = np.zeros(3, dtype=np.int64)
    for i, name in enumerate(names):
        info = sf.info(DATA / "wav" / name)
        # Resampling rounds the length by at most one sample; the frame count must agree.
        expected = len(target_grid(round(info.frames * 16000 / info.samplerate))[0])
        assert abs(frames[name] - expected) <= 1, (name, frames[name], expected)
        starts, ends = target_grid(8000 + 800 * (frames[name] - 1))
        length = cache["lengths"][i]
        raw = cache["scores"][i, :length]
        assert np.isfinite(raw).all() and np.isnan(cache["scores"][i, length:]).all()
        index = np.floor(((starts + ends) / 32000) / step_s).astype(int)
        target = activity_targets(annotations[name], starts, ends)
        counts += classification_counts(raw[index] >= 0.5, target).sum(axis=0)

    recorded = [metrics["results"]["raw"]["frame_micro"][k] for k in ("tp", "fp", "fn")]
    assert list(counts) == recorded, (list(counts), recorded)
    for mode in ("raw", median):
        rows = [r for r in per_file if r["mode"] == mode]
        assert len(rows) == 692
        totals = [sum(int(r[k]) for r in rows) for k in ("tp", "fp", "fn")]
        assert totals == [metrics["results"][mode]["frame_micro"][k] for k in ("tp", "fp", "fn")]
    print(f"PASS ({args.model}): 692 cached recordings, independent raw TP/FP/FN recount, "
          "per-recording totals, finite scores and padding.")
    for mode, result in metrics["results"].items():
        print(f"{mode}: frame F1 {result['frame_micro']['f1']:.4f}, event macro F1 "
              f"{result['event']['class_wise_average']['f_measure']['f_measure']:.4f}")


if __name__ == "__main__":
    main()
