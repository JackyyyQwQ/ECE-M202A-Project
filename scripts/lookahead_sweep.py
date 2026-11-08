"""Bounded-lookahead sweep for the 2023 reference detector (offline simulation).

    python scripts/lookahead_sweep.py --limit 20             # quick check
    python scripts/lookahead_sweep.py                        # validation set, L = 0, 0.5, 1, 2 s

Online, the BiGRU cannot see the future. For each 64 ms output frame t we run the
CNN+BiGRU on audio from the clip start up to L seconds after the end of frame t,
and keep only frame t's prediction, so every frame gets exactly L seconds of future
context ("full" = the whole available clip, the offline reference). Features are
normalized per window, as an online receiver would. No median filter is applied,
because a centered filter would add hidden lookahead. Threshold 0.5, no tuning.

Default data: the development (validation) set from scripts/fetch_validation.py.
`--dataset public` runs on the test set; use it only for frozen final reporting.
Writes output/lookahead/<dataset>/{summary.json, scores.npz}.
"""

import argparse
import csv
import json
import multiprocessing as mp
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import warnings

import numpy as np

# dcase_util (a sed_eval dependency) imports the deprecated pkg_resources API.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import sed_eval  # noqa: E402
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import pretrained_sed  # noqa: E402
from scripts.scoring import activity_targets, classification_counts, prf, target_grid  # noqa: E402
from scripts.pretrained_sed import CLASSES, decode_events, frame_intervals  # noqa: E402

RATE = 16000
FRAME = 1024              # samples per output frame: hop 256 x temporal pooling 4 = 64 ms
STEP_S = FRAME / RATE
MIN_WINDOW = 4 * FRAME    # very short windows get zero history prepended (no future)
DATASETS = {
    "validation": (ROOT / "data/raw/dcase2019_validation_youtube", "validation.tsv"),
    "public": (ROOT / "data/raw/dcase2019_public_eval", "public.tsv"),
}
_worker = {}


def init_worker(lookaheads, hop_frames):
    torch.set_num_threads(1)
    model, mel, _ = pretrained_sed.load_detector(ROOT / "cache/DESED_task", pretrained_sed.CHECKPOINT)
    _worker.update(model=model, mel=mel, lookaheads=lookaheads, hop=hop_frames, resamplers={})


def predict(wave):
    """Frame scores [frames, classes] for a 1-D 16 kHz waveform."""
    with torch.inference_mode():
        strong, _ = _worker["model"](pretrained_sed.normalized_mel(wave[None], _worker["mel"]))
    return strong[0].T.numpy()


def clip_scores(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.mean(axis=1))
    if rate != RATE:
        resampler = _worker["resamplers"].setdefault(rate, torchaudio.transforms.Resample(rate, RATE))
        wave = resampler(wave)
    usable = len(wave)
    start = time.perf_counter()
    full = predict(wave)
    frames, hop = len(full), _worker["hop"]
    result = {"full": full}
    for lookahead in _worker["lookaheads"]:
        scores = np.empty_like(full)
        future = int(round(lookahead * RATE))
        for first in range(0, frames, hop):
            last = min(first + hop, frames)
            end = min(usable, last * FRAME + future)   # audio available when frame `last-1` is reported
            window, offset = wave[:end], 0
            if end < MIN_WINDOW:                        # prepend silence; shifts frames by a whole number
                window, offset = torch.cat([torch.zeros(MIN_WINDOW), window]), MIN_WINDOW // FRAME
            scores[first:last] = predict(window)[offset + first:offset + last]
            if end == usable and offset == 0:  # whole clip visible: must equal the full-context run
                assert np.allclose(scores[first:last], full[first:last], atol=1e-5), "frame alignment error"
        result[lookahead] = scores
    return Path(path).name, usable, len(audio) / rate, result, time.perf_counter() - start


def threshold(raw, on=0.5):
    """No smoothing: active when the score is at least `on`."""
    return raw >= on


def hysteresis(raw, on=0.5, off=0.3):
    """Causal: start an event at score >= on, end it only when the score drops below off."""
    active = np.zeros(raw.shape, dtype=bool)
    state = np.zeros(raw.shape[1], dtype=bool)
    for t, frame in enumerate(raw):
        state = np.where(state, frame >= off, frame >= on)
        active[t] = state
    return active


def causal_mean(raw, frames=3, on=0.5):
    """Causal: mean of the current and previous frames (fewer at the clip start)."""
    total = np.cumsum(raw, axis=0)
    shifted = np.vstack([np.zeros((frames, raw.shape[1])), total[:-frames]]) if len(raw) > frames else 0
    count = np.minimum(np.arange(1, len(raw) + 1), frames)[:, None]
    return (total - shifted) / count >= on


# All decisions at frame t use frames <= t only: no delay beyond the lookahead L.
POSTPROCESSING = {"threshold 0.5": threshold, "hysteresis 0.5/0.3": hysteresis,
                  "causal mean (3 frames)": causal_mean}


def score(names, outputs, annotations, durations, usable, decide):
    """Frame (50 ms grid) and event metrics for one lookahead setting and decision rule."""
    counts = np.zeros((len(CLASSES), 3), dtype=np.int64)
    skipped, event_count, event_seconds = 0, 0, 0.0
    kinds = {"event": (True, True), "onset": (True, False), "offset": (False, True)}
    metrics = {k: sed_eval.sound_event.EventBasedMetrics(list(CLASSES), t_collar=0.2, percentage_of_length=0.2,
                                                       evaluate_onset=on, evaluate_offset=off)
               for k, (on, off) in kinds.items()}
    for name in names:
        active = decide(outputs[name])
        starts, ends = target_grid(usable[name])
        index = np.floor(((starts + ends) / (2 * RATE)) / STEP_S).astype(int)
        # Clips are not padded to 10 s, so the final partial 64 ms frame may have no
        # output; skip the (at most one) 50 ms target that falls in it.
        keep = index < len(active)
        skipped += int((~keep).sum())
        starts, ends, index = starts[keep], ends[keep], index[keep]
        counts += classification_counts(active[index], activity_targets(annotations[name], starts, ends))
        native_start, native_end = frame_intervals(len(active), STEP_S, durations[name])
        events = decode_events(active[:len(native_start)].astype(float), native_start, native_end, 0.5)
        predicted = [{"event_label": l, "event_onset": a, "event_offset": b} for l, a, b in events]
        event_count += len(events)
        event_seconds += sum(b - a for _, a, b in events)
        reference = [{"event_label": l, "event_onset": a, "event_offset": b} for l, a, b in annotations[name]]
        for metric in metrics.values():
            metric.evaluate(reference, predicted)
    result = {"frame_micro": prf(counts.sum(axis=0)), "skipped_targets": skipped,
              "predicted_events": event_count,
              "reference_events": sum(len(annotations[n]) for n in names),
              "mean_predicted_event_s": event_seconds / event_count if event_count else 0.0}
    for kind, metric in metrics.items():
        results = metric.results()
        result[f"{kind}_macro_f1"] = results["class_wise_average"]["f_measure"]["f_measure"]
        result[f"{kind}_micro_f1"] = results["overall"]["f_measure"]["f_measure"]
    return result


def save_scores(path, names, outputs, usable, durations, compute):
    keys = list(outputs)
    frames = max(len(outputs[k][n]) for k in keys for n in names)
    scores = np.full((len(keys), len(names), frames, len(CLASSES)), np.nan, dtype=np.float32)
    for i, key in enumerate(keys):
        for j, name in enumerate(names):
            scores[i, j, :len(outputs[key][name])] = outputs[key][name]
    np.savez_compressed(path, keys=np.array([str(k) for k in keys]), names=np.array(names), scores=scores,
                        lengths=np.array([len(outputs[keys[0]][n]) for n in names]),
                        usable=np.array([usable[n] for n in names]),
                        durations=np.array([durations[n] for n in names]), compute=np.array(compute))


def load_scores(path):
    data = np.load(path)
    names = [str(n) for n in data["names"]]
    keys = ["full" if k == "full" else float(k) for k in data["keys"]]
    outputs = {key: {n: data["scores"][i, j, :data["lengths"][j]] for j, n in enumerate(names)}
               for i, key in enumerate(keys)}
    usable = dict(zip(names, data["usable"].tolist()))
    durations = dict(zip(names, data["durations"].tolist()))
    return names, outputs, usable, durations, data["compute"].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="validation")
    parser.add_argument("--lookaheads", default="0,0.5,1,2", help="seconds, comma-separated")
    parser.add_argument("--hop-frames", type=int, default=1,
                        help="frames reported per model run; >1 is faster but gives frames L..L+(hop-1)*64 ms")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--limit", type=int, help="use only the first N recordings (quick check)")
    parser.add_argument("--rescore", action="store_true",
                        help="reuse the saved scores.npz; only recompute metrics (no model runs)")
    args = parser.parse_args()
    folder, labels = DATASETS[args.dataset]
    out = ROOT / "output/lookahead" / args.dataset
    if args.dataset == "public":
        print("WARNING: public evaluation is the test set; do not tune on these results.", flush=True)

    annotations = defaultdict(list)
    with (folder / labels).open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            annotations[row["filename"]]  # recordings without events still count
            if row["event_label"]:
                annotations[row["filename"]].append(
                    (row["event_label"], float(row["onset"]), float(row["offset"])))

    if args.rescore:
        names, outputs, usable, durations, compute = load_scores(out / "scores.npz")
        lookaheads = [k for k in outputs if k != "full"]
        print(f"Rescoring saved scores: {len(names)} recordings; lookaheads {lookaheads} s", flush=True)
    else:
        lookaheads = [float(x) for x in args.lookaheads.split(",")]
        names = sorted(n for n in annotations if (folder / "wav" / n).exists())
        if args.limit:
            names = names[:args.limit]
        if not names:
            raise SystemExit(f"No recordings found in {folder / 'wav'}")
        print(f"{args.dataset}: {len(names)} of {len(annotations)} labelled recordings; "
              f"lookaheads {lookaheads} s; {args.workers} workers", flush=True)
        outputs = {key: {} for key in ["full", *lookaheads]}
        durations, usable, compute = {}, {}, []
        with mp.get_context("spawn").Pool(args.workers, init_worker, (lookaheads, args.hop_frames)) as pool:
            paths = [folder / "wav" / n for n in names]
            for i, (name, samples, duration, result, seconds) in enumerate(pool.imap_unordered(clip_scores, paths)):
                usable[name], durations[name] = samples, duration
                compute.append(seconds)
                for key, scores in result.items():
                    outputs[key][name] = scores
                if (i + 1) % 25 == 0 or i + 1 == len(names):
                    print(f"{i + 1}/{len(names)} recordings", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        save_scores(out / "scores.npz", names, outputs, usable, durations, compute)

    rows = []
    for rule, decide in POSTPROCESSING.items():
        for key in ["full", *lookaheads]:
            result = score(names, outputs[key], annotations, durations, usable, decide)
            rows.append({"postprocessing": rule, "lookahead_s": None if key == "full" else key, **result})
    summary = {
        "dataset": args.dataset, "recordings": len(names), "labelled_recordings": len(annotations),
        "detector": pretrained_sed.CHECKPOINT.name, "hop_frames": args.hop_frames, "frame_s": STEP_S,
        "protocol": "Each 64 ms frame predicted from audio up to L s after its end; per-window "
                    "feature normalization; frame metrics on the 50 ms grid; sed_eval event metrics "
                    "(onset collar 200 ms; offset max(200 ms, 20%)) on the native grid. Post-processing "
                    "is causal (frames <= t only); no centered median filter.",
        "latency_note": "Added reporting delay is L plus the 64 ms frame; computation and transport excluded.",
        "median_compute_s_per_recording": statistics.median(compute),
        "results": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    current = None
    for row in rows:
        if row["postprocessing"] != current:
            current = row["postprocessing"]
            print(f"\n[{current}]  (event/onset/offset F1 are class-macro, as in DCASE; "
                  f"'event uF1' pools all classes)\n{'lookahead':>10} {'frame F1':>9} {'event F1':>9}"
                  f" {'event uF1':>10} {'onset F1':>9} {'offset F1':>10} {'events':>7} {'mean s':>7}")
        label = "full" if row["lookahead_s"] is None else f"{row['lookahead_s']:g} s"
        print(f"{label:>10} {row['frame_micro']['f1']:9.4f} {row['event_macro_f1']:9.4f} "
              f"{row['event_micro_f1']:10.4f} {row['onset_macro_f1']:9.4f} {row['offset_macro_f1']:10.4f}"
              f" {row['predicted_events']:7d} {row['mean_predicted_event_s']:7.2f}")
    print(f"\nReference: {rows[0]['reference_events']} events in {len(names)} recordings; "
          f"median compute {summary['median_compute_s_per_recording']:.1f} s per recording")
    print(f"Saved {out / 'summary.json'}")


if __name__ == "__main__":
    main()
