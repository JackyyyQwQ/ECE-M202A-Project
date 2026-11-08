"""Evaluate the frozen DCASE CNN+BiGRU on a fixed labeled recording manifest.

Primary result: raw scores, threshold 0.5, no fitted thresholds or model updates.
The 50 ms evaluation grid matches the custom baseline's target intervals, but
this pilot is not its original split and the BiGRU has full-window context.
"""

import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment

from scripts.pretrained_sed import (
    CHECKPOINT, CLASSES, ROOT, UPSTREAM_COMMIT, decode_events, load_detector,
    normalized_mel, write_tsv,
)


def target_grid(sample_count, sample_rate=16000):
    ends = np.arange(8000, sample_count + 1, 800)
    return ends - 800, ends


def nominal_ten_seconds(audio, sample_rate):
    """Handle observed one/two-sample duration rounding; reject larger mismatches."""
    expected = 10 * sample_rate
    difference = len(audio) - expected
    if abs(difference) > 2 or not np.isfinite(audio).all():
        raise ValueError("Expected finite audio within two samples of ten seconds")
    audio = audio[:expected]
    if len(audio) < expected:
        audio = np.pad(audio, ((0, expected - len(audio)), (0, 0)))
    return audio, difference


def activity_targets(events, starts, ends, sample_rate=16000):
    labels = np.zeros((len(starts), len(CLASSES)), dtype=bool)
    for label, onset, offset in events:
        active = (onset * sample_rate < ends) & (offset * sample_rate > starts)
        labels[active, CLASSES.index(label)] = True
    return labels


def classification_counts(predicted, reference):
    return np.stack((
        (predicted & reference).sum(axis=0),
        (predicted & ~reference).sum(axis=0),
        (~predicted & reference).sum(axis=0),
    ), axis=-1)


def prf(counts):
    tp, fp, fn = map(int, counts)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def overlap_matches(predicted, reference):
    """Same-class one-to-one matching: maximize pair count, then total IoU."""
    if not predicted or not reference:
        return []
    utility = np.zeros((len(predicted), len(reference)))
    bonus = min(utility.shape) + 1
    for i, (label, onset, offset) in enumerate(predicted):
        for j, (other, begin, end) in enumerate(reference):
            intersection = min(offset, end) - max(onset, begin)
            if label == other and intersection > 0:
                utility[i, j] = bonus + intersection / (max(offset, end) - min(onset, begin))
    rows, cols = linear_sum_assignment(utility, maximize=True)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if utility[i, j] > 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/dcase2019")
    parser.add_argument("--output", type=Path, default=ROOT / "output/pretrained_pilot")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--all", action="store_true", help="Use full_manifest.csv, preserving the pilot")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.threads < 1:
        parser.error("threshold must be in [0, 1]; threads must be positive")
    prefix = "full" if args.all else "pilot"
    if args.all and args.output == ROOT / "output/pretrained_pilot":
        args.output = ROOT / "output/pretrained_full"
    manifest_path = args.data / f"{prefix}_manifest.csv"
    with manifest_path.open() as handle:
        manifest = list(csv.DictReader(handle))
    if not manifest or len({r["filename"] for r in manifest}) != len(manifest):
        raise ValueError("Manifest must be non-empty with unique filenames")
    if args.all and len(manifest) != 2045:
        raise ValueError("Full evaluation requires all 2045 recordings")
    annotations = defaultdict(list)
    with (args.data / "synthetic_dataset.csv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            label, onset, offset = row["event_label"], float(row["onset"]), float(row["offset"])
            if label not in CLASSES or not 0 <= onset < offset <= 10:
                raise ValueError(f"Unexpected annotation: {row}")
            annotations[row["filename"]].append((label, onset, offset))
    if any(r["filename"] not in annotations for r in manifest):
        raise ValueError("Every selected recording must have annotations")
    torch.set_num_threads(args.threads)
    model, mel, config = load_detector(ROOT / "cache/DESED_task", args.checkpoint)
    rate = config["feats"]["sample_rate"]
    step_s = config["feats"]["hop_length"] * config["data"]["net_subsample"] / rate
    totals = {name: {"frames": np.zeros((len(CLASSES), 3), dtype=np.int64),
                     "events": np.zeros(3, dtype=np.int64), "onset": [], "offset": []}
              for name in ("raw", "median7")}
    resamplers, compute_ms, cached_scores, event_rows, recording_rows = {}, [], [], [], []
    total_frames, total_seconds = 0, 0.0
    adjusted_recordings = []
    source_channels, source_rates = Counter(), Counter()
    started = time.perf_counter()
    with torch.inference_mode():
        model(normalized_mel(torch.zeros(1, 160000), mel))  # one unmeasured warm-up
        for index, item in enumerate(manifest):
            path = args.data / "wav" / item["filename"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Audio hash mismatch: {path.name}")
            audio, original_rate = sf.read(path, dtype="float32", always_2d=True)
            source_channels[audio.shape[1]] += 1
            source_rates[original_rate] += 1
            audio, difference = nominal_ten_seconds(audio, original_rate)
            if difference:
                adjusted_recordings.append({"filename": path.name, "extra_source_samples": difference})
            waveform = torch.from_numpy(audio.mean(axis=1))
            if original_rate != rate:
                if original_rate not in resamplers:
                    resamplers[original_rate] = torchaudio.transforms.Resample(original_rate, rate)
                waveform = resamplers[original_rate](waveform)
            start = time.perf_counter()
            scores, _ = model(normalized_mel(waveform.unsqueeze(0), mel))
            compute_ms.append((time.perf_counter() - start) * 1000)
            if not torch.isfinite(scores).all():
                raise RuntimeError(f"Non-finite prediction: {path.name}")
            raw = scores[0].T.numpy()
            cached_scores.append(raw)
            starts, ends = target_grid(len(waveform))
            targets = activity_targets(annotations[path.name], starts, ends)
            midpoints_s = (starts + ends) / (2 * rate)
            frame_indices = np.floor(midpoints_s / step_s).astype(int)
            if frame_indices.max() >= len(raw):
                raise RuntimeError("Model output does not cover evaluation frame midpoints")
            begin_s, end_s = starts / rate, ends / rate
            reference = [event for event in annotations[path.name]
                         if event[2] > begin_s[0] and event[1] < end_s[-1]]
            for name, values in (("raw", raw), ("median7", median_filter(raw, size=(7, 1)))):
                predicted = values[frame_indices] >= args.threshold
                counts = classification_counts(predicted, targets)
                totals[name]["frames"] += counts
                # Decode the same thresholded 50 ms activity used for frame scoring.
                events = decode_events(predicted.astype(float), begin_s, end_s, 0.5)
                pairs = overlap_matches(events, reference)
                totals[name]["events"] += (len(pairs), len(events) - len(pairs), len(reference) - len(pairs))
                for i, j in pairs:
                    if reference[j][1] > begin_s[0]:
                        totals[name]["onset"].append(abs(events[i][1] - reference[j][1]) * 1000)
                    if reference[j][2] < end_s[-1]:
                        totals[name]["offset"].append(abs(events[i][2] - reference[j][2]) * 1000)
                event_rows.extend((path.name, name, *event) for event in events)
                recording_rows.append((path.name, name, len(starts), *map(int, counts.sum(axis=0)),
                                       len(events), len(reference), len(pairs), compute_ms[-1]))
            total_frames += len(starts)
            total_seconds += len(waveform) / rate
            if (index + 1) % (128 if args.all else 8) == 0:
                print(f"Evaluated {index + 1}/{len(manifest)} recordings", flush=True)
    report = {
        "dataset": "DCASE2019 synthetic, Zenodo 2583796", "mode": "offline_full_window",
        "selection": json.loads((args.data / f"{prefix}_source.json").read_text()),
        "recordings": len(manifest), "audio_seconds": total_seconds, "frames": total_frames,
        "classes": CLASSES, "threshold": args.threshold, "training_performed": False,
        "duration_rounding": adjusted_recordings,
        "source_audio_channels": dict(source_channels),
        "source_audio_sample_rates": dict(source_rates),
        "selected_annotation_events": sum(len(annotations[r["filename"]]) for r in manifest),
        "threshold_tuned": False, "original_32_recording_validation_split": False,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint": args.checkpoint.name, "upstream_commit": UPSTREAM_COMMIT,
        "timing": {"device": "cpu", "threads": args.threads, "platform": platform.platform(),
                   "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                   "warm_compute_ms": compute_ms, "median_compute_ms": statistics.median(compute_ms),
                   "p95_compute_ms": float(np.quantile(compute_ms, 0.95)),
                   "total_run_s": time.perf_counter() - started,
                   "scope": "compute = mel/dB/normalization/model; excludes I/O/resampling/postprocessing/waiting/transport"},
        "evaluation": {
            "frame_grid": "191 intervals [0.45,0.50), ..., [9.95,10.00); any-overlap ground truth",
            "score_alignment": "Use the official 64 ms score bin containing each 50 ms target midpoint",
            "event_matching": "Same class, positive overlap, maximum pair count then IoU; permissive, not collar F1 or PSDS",
            "boundary_mae": "Conditional on overlap-matched pairs; exclude onsets <=0.45 s and offsets >=10 s; unmatched events counted in F1",
            "limitations": "Public synthetic training recordings, not the old validation split. Full future context and full-window normalization. Pretraining-data overlap not audited; not an independent generalization benchmark or causal-vs-bidirectional ablation.",
        }, "results": {},
    }
    class_rows = []
    for name, data in totals.items():
        metrics = {"frame_micro": prf(data["frames"].sum(axis=0)),
                   "event_overlap_micro": prf(data["events"]), "per_class": {}}
        for label, counts in zip(CLASSES, data["frames"]):
            metrics["per_class"][label] = prf(counts)
            class_rows.append((name, label, *map(int, counts), *[prf(counts)[k] for k in ("precision", "recall", "f1")]))
        metrics["frame_macro_f1"] = statistics.mean(v["f1"] for v in metrics["per_class"].values())
        for boundary in ("onset", "offset"):
            values = data[boundary]
            metrics[boundary + "_mae_ms"] = statistics.mean(values) if values else None
            metrics[boundary + "_matched_count"] = len(values)
        report["results"][name] = metrics
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    write_tsv(args.output / "per_class.tsv", ("mode", "class", "tp", "fp", "fn", "precision", "recall", "f1"), class_rows)
    write_tsv(args.output / "events.tsv", ("filename", "mode", "event_label", "onset", "offset"), event_rows)
    write_tsv(args.output / "per_recording.tsv", ("filename", "mode", "frames", "tp", "fp", "fn", "predicted_events", "reference_events", "matches", "compute_ms"), recording_rows)
    np.savez_compressed(args.output / "scores.npz", filenames=[r["filename"] for r in manifest], scores=np.stack(cached_scores), classes=CLASSES)
    for name, metrics in report["results"].items():
        print(f"{name}: frame micro F1={metrics['frame_micro']['f1']:.2%}, event overlap F1={metrics['event_overlap_micro']['f1']:.2%}")
    print(f"Saved {args.output}; this is not the old 32-recording comparison")


if __name__ == "__main__":
    main()
