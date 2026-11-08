"""Evaluate a frozen official CNN+BiGRU on the DCASE2019 public evaluation set.

    python scripts/evaluate_public.py --model 2023    # or --model 2019

Run from the repository root with the project conda environment. Inputs: the 692
WAVs and manifest from scripts/fetch_public_eval.py, the model checkpoint and its
pinned upstream checkout (see docs/dcase2019_model_comparison.md). Writes
output/public_eval/<model>/{metrics.json,scores.npz,events.tsv,per_recording.tsv}.

No training or threshold tuning (threshold 0.5). Each model sees the full clip,
i.e. unlimited future context: these are offline references, not online results.
"""

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sed_eval
import soundfile as sf
import torch
import torchaudio
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dcase2019_model, pretrained_sed  # noqa: E402
from scripts.evaluate_pretrained import (  # noqa: E402
    activity_targets, classification_counts, prf, target_grid)
from scripts.pretrained_sed import CLASSES, decode_events, frame_intervals, write_tsv  # noqa: E402

DATA = ROOT / "data/raw/dcase2019_public_eval"
# Official 2019 metadata, identical to the archive's public.tsv (checked below).
METADATA = dcase2019_model.UPSTREAM / "dataset/metadata"
TARGET_HOP = 800  # 50 ms scoring grid at 16 kHz (scripts/evaluate_pretrained.py)


def model_2023():
    model, mel, _ = pretrained_sed.load_detector(ROOT / "cache/DESED_task", pretrained_sed.CHECKPOINT)
    return {
        "model": model, "rate": 16000, "step_s": 0.064, "median": 7, "pad_to": 160000,
        "dtype": "float32", "checkpoint": pretrained_sed.CHECKPOINT,
        "upstream_commit": pretrained_sed.UPSTREAM_COMMIT,
        "features": lambda wave: pretrained_sed.normalized_mel(wave, mel),
        "scores": lambda strong: strong[0].T.numpy(),  # [classes, frames] -> [frames, classes]
        "provenance": {
            "checkpoint_source": "https://zenodo.org/records/7759146",
            "frontend": "16 kHz; 128 mel; Hamming 2048, hop 256; amplitude dB clamp [-50, 80]; "
                        "per-clip minmax to [0, 1] (2023 training-code TorchScaler, commit "
                        + pretrained_sed.SCALER_REFERENCE_COMMIT + ")",
            "padding": "Waveform zero-padded to 10 s; padded region not scored",
        },
    }


def model_2019():
    model, scaler = dcase2019_model.load_detector()
    return {
        "model": model, "rate": dcase2019_model.SAMPLE_RATE, "step_s": dcase2019_model.STEP_S,
        "median": 5, "pad_to": 0, "dtype": "float64", "checkpoint": dcase2019_model.CHECKPOINT,
        "upstream_commit": dcase2019_model.UPSTREAM_COMMIT,
        "features": lambda wave: dcase2019_model.normalized_mel(wave, scaler),
        "scores": lambda strong: strong[0].numpy(),  # already [frames, classes]
        "provenance": {
            "checkpoint_source": dcase2019_model.CHECKPOINT_SOURCE, "epoch": 91,
            "frontend": "44.1 kHz; symmetric Hamming 2048, hop 511; magnitude mel 64, htk=False, "
                        "norm=None; amplitude_to_db; checkpoint training mean/std",
            "padding": "Log features zero-padded to >= 864 frames; 11 s clip padded to a "
                       "multiple of 8, never truncated",
        },
    }


def video_ids(path):
    """YouTube IDs in an official metadata file (names like Y<id>_<start>_<end>.wav)."""
    ids = set()
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            video = Path(row["filename"]).stem.rsplit("_", 2)[0]
            ids.add(video[1:] if len(video) == 12 and video.startswith("Y") else video)
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("2019", "2023"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    spec = model_2023() if args.model == "2023" else model_2019()
    model, rate, step_s = spec["model"], spec["rate"], spec["step_s"]
    out = ROOT / "output/public_eval" / args.model

    manifest = json.loads((DATA / "manifest.json").read_text())
    files = manifest["files"]
    assert len(files) == 692
    with (DATA / "public.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    with (METADATA / "eval/public.tsv").open() as handle:
        assert rows == list(csv.DictReader(handle, delimiter="\t")), "Archive and official annotations differ"
    annotations = defaultdict(list)
    for row in rows:
        assert row["event_label"] in CLASSES
        annotations[row["filename"]].append((row["event_label"], float(row["onset"]), float(row["offset"])))
    assert set(annotations) == {item["filename"] for item in files}

    # Recordings whose source video also appears in the published 2019 training lists.
    corpus = video_ids(METADATA / "eval/public.tsv")
    training = {"weak": video_ids(METADATA / "train/weak.tsv"),
                "unlabel": video_ids(METADATA / "train/unlabel_in_domain.tsv")}
    known = set().union(*(corpus & ids for ids in training.values()))
    overlap_files = {item["filename"] for item in files
                     if Path(item["filename"]).stem.rsplit("_", 2)[0] in known}

    median = f"median{spec['median']}"
    modes = ("raw", median, "raw_no_known_overlap", f"{median}_no_known_overlap")
    totals = {mode: np.zeros((len(CLASSES), 3), dtype=np.int64) for mode in modes}
    evaluators = {mode: {
        "event": sed_eval.sound_event.EventBasedMetrics(list(CLASSES), t_collar=0.2, percentage_of_length=0.2),
        "segment": sed_eval.sound_event.SegmentBasedMetrics(list(CLASSES), time_resolution=1.0),
    } for mode in modes}
    timings, cached, event_rows, per_file, adjustments = [], [], [], [], []
    channels, rates, resamplers, frame_count, duration_total = Counter(), Counter(), {}, 0, 0.0

    with torch.inference_mode():
        model(spec["features"](torch.zeros(1, 10 * rate)))  # unmeasured warm-up
        for n, item in enumerate(files):
            path = DATA / "wav" / item["filename"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
            audio, source_rate = sf.read(path, dtype=spec["dtype"], always_2d=True)
            assert len(audio) and np.isfinite(audio).all()
            channels[audio.shape[1]] += 1
            rates[source_rate] += 1
            duration = len(audio) / source_rate
            duration_total += duration
            if len(audio) != 10 * source_rate:
                adjustments.append({"filename": path.name, "source_samples": len(audio),
                                    "difference_from_10s_samples": len(audio) - 10 * source_rate})
            wave = torch.from_numpy(audio.mean(axis=1))  # channel average, as upstream to_mono
            if source_rate != rate:
                resamplers.setdefault(source_rate, torchaudio.transforms.Resample(source_rate, rate))
                wave = resamplers[source_rate](wave)
            usable = len(wave)  # model-rate samples of real audio
            wave = torch.nn.functional.pad(wave, (0, max(0, spec["pad_to"] - usable)))

            start = time.perf_counter()
            strong, _ = model(spec["features"](wave[None]))
            timings.append((time.perf_counter() - start) * 1000)
            raw = spec["scores"](strong)
            assert raw.ndim == 2 and raw.shape[1] == len(CLASSES) and np.isfinite(raw).all()
            cached.append(raw)

            # 50 ms frame targets (16 kHz sample indices); each takes the native
            # prediction bin containing its midpoint.
            starts, ends = target_grid(int(round(usable * 16000 / rate)))
            target = activity_targets(annotations[path.name], starts, ends)
            index = np.floor(((starts + ends) / 32000) / step_s).astype(int)
            assert index.max() < len(raw)
            frame_count += len(starts)
            # Events use the native grid over the full available clip.
            native_start, native_end = frame_intervals(len(raw), step_s, duration)
            reference = [{"event_label": l, "event_onset": a, "event_offset": b}
                         for l, a, b in annotations[path.name]]
            for mode, scores in (("raw", raw),
                                 (median, median_filter(raw, size=(spec["median"], 1), mode="reflect"))):
                counts = classification_counts(scores[index] >= 0.5, target)
                events = decode_events((scores[:len(native_start)] >= 0.5).astype(float),
                                       native_start, native_end, 0.5)
                predicted = [{"event_label": l, "event_onset": a, "event_offset": b} for l, a, b in events]
                groups = [mode] + ([] if path.name in overlap_files else [f"{mode}_no_known_overlap"])
                for group in groups:
                    totals[group] += counts
                    evaluators[group]["event"].evaluate(reference, predicted)
                    evaluators[group]["segment"].evaluate(reference, predicted, evaluated_length_seconds=duration)
                event_rows.extend((path.name, mode, *event) for event in events)
                per_file.append((path.name, mode, len(starts), *map(int, counts.sum(axis=0)),
                                 len(events), len(reference), timings[-1]))
            if (n + 1) % 32 == 0:
                print(f"Public eval ({args.model}): {n + 1}/692", flush=True)

    checkpoint = spec["checkpoint"]
    report = {
        "dataset": "DCASE2019 Task 4 public evaluation (real audio)", "record": 4560759,
        "archive_md5": manifest["archive_md5"], "audio_files": len(files), "annotation_events": len(rows),
        "model_year": int(args.model), "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "upstream_commit": spec["upstream_commit"], **spec["provenance"],
        "training_performed": False, "threshold_tuned": False, "threshold": 0.5,
        "coverage": "All 692 WAVs; no quality-based exclusions; complete 11 s clip retained",
        "metadata_sha256": hashlib.sha256((DATA / "public.tsv").read_bytes()).hexdigest(),
        "source_channels": dict(channels), "source_sample_rates": dict(rates),
        "original_duration_seconds": duration_total, "scored_50ms_frames": frame_count,
        "duration_adjustments": adjustments,
        "published_2019_training_video_id_overlap": {k: len(corpus & v) for k, v in training.items()},
        "known_overlap_files": sorted(overlap_files),
        "recordings_excluding_known_overlap": len(files) - len(overlap_files),
        "overlap_audit_limit": "Published 2019 weak/unlabeled lists only; does not prove training independence",
        "frame_protocol": f"Any-overlap 50 ms targets from 0.45 s; native {step_s * 1000:.2f} ms bin at target midpoint",
        "event_protocol": "sed_eval optimal matching; onset collar 200 ms; offset max(200 ms, 20% of reference duration)",
        "segment_protocol": "sed_eval one-second segments",
        "timing": {"device": "CPU", "threads": args.threads, "platform": platform.platform(),
                   "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                   "median_compute_ms": statistics.median(timings),
                   "p95_compute_ms": float(np.quantile(timings, 0.95)),
                   "per_recording_compute_ms": timings,
                   "scope": "features + normalization + CNN/BiGRU/head; excludes loading, "
                            "resampling, postprocessing, I/O, audio waiting and transport"},
        "limitations": "Offline full-context detector; no MCU, communication or deadline test.",
        "results": {},
    }
    for mode, counts in totals.items():
        report["results"][mode] = {
            "frame_micro": prf(counts.sum(axis=0)),
            "frame_macro_f1": statistics.mean(prf(v)["f1"] for v in counts),
            "per_class_frame": {c: prf(v) for c, v in zip(CLASSES, counts)},
            "event": evaluators[mode]["event"].results(),
            "segment": evaluators[mode]["segment"].results(),
        }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    padded = np.full((len(cached), max(len(x) for x in cached), len(CLASSES)), np.nan, dtype=np.float32)
    for i, scores in enumerate(cached):
        padded[i, :len(scores)] = scores
    np.savez_compressed(out / "scores.npz", filenames=[item["filename"] for item in files],
                        scores=padded, lengths=[len(x) for x in cached], classes=CLASSES)
    write_tsv(out / "events.tsv", ("filename", "mode", "event_label", "onset", "offset"), event_rows)
    write_tsv(out / "per_recording.tsv", ("filename", "mode", "frames", "tp", "fp", "fn",
                                          "predicted_events", "reference_events", "compute_ms"), per_file)
    print(f"Finished 692 recordings, {frame_count} frames; median compute "
          f"{report['timing']['median_compute_ms']:.1f} ms")
    for mode, result in report["results"].items():
        print(f"{mode}: frame F1 {result['frame_micro']['f1']:.4f}, event macro F1 "
              f"{result['event']['class_wise_average']['f_measure']['f_measure']:.4f}")


if __name__ == "__main__":
    main()
