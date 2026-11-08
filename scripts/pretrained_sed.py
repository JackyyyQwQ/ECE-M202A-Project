"""Offline inference with the official DCASE 2023 CNN+BiGRU student.

Uses upstream model code without training; feature scaling follows the 2023
training code (see normalized_mel), not the pinned 2024 scaler.
This is a full-window reference, not a streaming or deadline-aware detector.
"""

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.ndimage import median_filter


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "c6bcb45b8b986ccde5c56bb86eefaf9d19b2320c"
CHECKPOINT = ROOT / "checkpoints/dcase2023/legacy_epoch=176-step=20886.ckpt"
CLASSES = (
    "Alarm_bell_ringing", "Blender", "Cat", "Dishes", "Dog",
    "Electric_shaver_toothbrush", "Frying", "Running_water", "Speech",
    "Vacuum_cleaner",
)


def load_detector(repo, checkpoint):
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f"Use upstream revision {UPSTREAM_COMMIT}; see setup guide.")
    if subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("Upstream checkout has local changes; use a clean checkout.")
    sys.path.insert(0, str(repo))
    from desed_task.nnet.CRNN import CRNN

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["hyper_parameters"]
    if config["net"].get("use_embeddings", False) or config["net"]["nclass"] != 10:
        raise ValueError("Use the plain 10-class legacy checkpoint, without BEATs.")
    model = CRNN(**config["net"])
    # Upstream profiling added counters, which are not learned model parameters.
    weights = {k: v for k, v in saved["sed_student"].items()
               if k.split(".")[-1] not in {"total_ops", "total_params"}}
    model.load_state_dict(weights, strict=True)
    model.eval()  # Upstream train()/eval() does not return the model.

    feats = config["feats"]
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=feats["sample_rate"], n_fft=feats["n_window"],
        win_length=feats["n_window"], hop_length=feats["hop_length"],
        f_min=feats["f_min"], f_max=feats["f_max"], n_mels=feats["n_mels"],
        window_fn=torch.hamming_window, wkwargs={"periodic": False}, power=1,
    )
    # Restore the exact saved Hamming window and mel-filter matrix.
    mel.load_state_dict({k.removeprefix("mel_spec."): v
                         for k, v in saved["state_dict"].items()
                         if k.startswith("mel_spec.")}, strict=True)
    scaler = config["scaler"]
    if (scaler["statistic"], scaler["normtype"], scaler["dims"]) != (
        "instance", "minmax", [1, 2]
    ):
        raise ValueError("This runner implements the legacy instance-minmax scaler.")
    return model, mel, config


# Commit whose desed_task/utils/scaler.py matches the 2023 checkpoint's training code.
SCALER_REFERENCE_COMMIT = "9f9b610fbcdca0996ac38dc4b8c2f2e0e18e25f0"


def normalized_mel(audio, mel):
    """Log-mel features exactly as the 2023 checkpoint was trained on.

    Matches the 2023 recipe's SEDTask4.take_log (amin 1e-5, clamp [-50, 80] dB)
    and the March 2023 TorchScaler instance-minmax, which maps each clip to
    [0, 1]. The pinned 2024 upstream checkout (UPSTREAM_COMMIT) later changed
    that scaler to return `out * 2 - 1`, i.e. [-1, 1]. Using the 2024 range with
    this 2023 checkpoint shifts every input and suppresses detections, especially
    of stationary classes (vacuum cleaner, blender, running water). Keep [0, 1]
    unless the checkpoint changes. See SCALER_REFERENCE_COMMIT.
    """
    to_db = torchaudio.transforms.AmplitudeToDB(stype="amplitude")
    to_db.amin = 1e-5  # As in the recipe; no numeric effect after the -50 dB clamp.
    features = to_db(mel(audio)).clamp(min=-50, max=80)
    low = features.amin(dim=(1, 2), keepdim=True)
    high = features.amax(dim=(1, 2), keepdim=True)
    return (features - low) / (high - low + 1e-8)


def frame_intervals(count, step_s, duration_s):
    """Official hop-based time grid, clipped to real audio rather than padding."""
    starts = np.arange(count) * step_s
    starts = starts[starts < duration_s]
    return starts, np.minimum(starts + step_s, duration_s)


def decode_events(scores, starts, ends, threshold):
    events = []
    for index, label in enumerate(CLASSES):
        active = scores[:, index] > threshold
        changes = np.diff(np.r_[False, active, False].astype(np.int8))
        for first, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            events.append((label, float(starts[first]), float(ends[stop - 1])))
    return sorted(events, key=lambda event: (event[1], event[0]))


def write_tsv(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="One recording, at most 10 seconds")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--upstream", type=Path, default=ROOT / "cache/DESED_task")
    parser.add_argument("--output", type=Path, default=ROOT / "output/pretrained_sed")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--median-frames", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.repeats < 1 or args.threads < 1:
        parser.error("threshold must be in [0, 1]; repeats and threads must be positive")
    if args.median_frames < 1 or args.median_frames % 2 != 1:
        parser.error("median-frames must be a positive odd number")
    torch.set_num_threads(args.threads)
    model, mel, config = load_detector(args.upstream.resolve(), args.checkpoint)
    audio, original_rate = sf.read(args.audio, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        parser.error("audio must contain non-empty, finite samples")
    duration_s = len(audio) / original_rate
    sample_rate = config["feats"]["sample_rate"]
    window_s = config["data"]["audio_max_len"]
    if duration_s > window_s:
        parser.error("split recordings longer than 10 seconds before using this offline runner")
    waveform = torch.from_numpy(audio.mean(axis=1))
    if original_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, original_rate, sample_rate)
    window_samples = int(window_s * sample_rate)
    waveform = torch.nn.functional.pad(waveform, (0, window_samples - len(waveform)))
    waveform = waveform.unsqueeze(0)

    with torch.inference_mode():
        start = time.perf_counter()
        scores, _ = model(normalized_mel(waveform, mel))
        first_ms = (time.perf_counter() - start) * 1000
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            scores, _ = model(normalized_mel(waveform, mel))
            timings.append((time.perf_counter() - start) * 1000)
    if not torch.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise RuntimeError("invalid detector output")
    raw_scores = scores[0].T.numpy()
    step_s = config["feats"]["hop_length"] * config["data"]["net_subsample"] / sample_rate
    starts, ends = frame_intervals(len(raw_scores), step_s, duration_s)
    # Filter the full padded model output, then exclude timestamps beyond real audio.
    filtered = median_filter(raw_scores, size=(args.median_frames, 1), mode="reflect")
    events = decode_events(filtered[:len(starts)], starts, ends, args.threshold)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in (("scores_raw.tsv", raw_scores), ("scores_filtered.tsv", filtered)):
        write_tsv(args.output / name, ("onset", "offset", *CLASSES),
                  ((float(a), float(b), *map(float, row))
                   for a, b, row in zip(starts, ends, values)))
    write_tsv(args.output / "events.tsv", ("filename", "event_label", "onset", "offset"),
              ((args.audio.name, *event) for event in events))
    report = {
        "mode": "offline_full_window_cpu", "training_performed": False,
        "upstream_commit": UPSTREAM_COMMIT, "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "audio": str(args.audio.resolve()), "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "input_sample_rate": original_rate, "model_sample_rate": sample_rate,
        "duration_s": duration_s, "model_window_s": window_s,
        "classes": CLASSES, "model_output_shape": list(scores.shape),
        "frame_step_s": step_s, "exported_frames": len(starts),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "threshold": args.threshold, "median_frames": args.median_frames,
        "device": "cpu", "threads": args.threads, "platform": platform.platform(),
        "machine": platform.machine(), "torch": torch.__version__,
        "torchaudio": torchaudio.__version__, "first_compute_ms": first_ms,
        "warm_compute_ms": timings, "warm_median_compute_ms": statistics.median(timings),
        "timing_scope": "mel + dB + normalization + CNN/BiGRU/head; excludes load, decode, resampling, event decoding, I/O, audio waiting and transport",
        "event_count": len(events), "accuracy": None,
        "limitations": "Unlabeled offline smoke run; full-window future context, centered STFT, instance normalization and symmetric median filtering. Not online latency or benchmark reproduction.",
    }
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Loaded official CNN+BiGRU: {report['parameter_count']:,} parameters; no training")
    print(f"Output {list(scores.shape)}; frame step {step_s * 1000:g} ms")
    print(f"Warm median compute: {report['warm_median_compute_ms']:.1f} ms (CPU, {args.threads} threads)")
    for label, onset, offset in events:
        print(f"{label}: {onset:.3f}–{offset:.3f} s")
    print(f"Saved {args.output.resolve()}; accuracy not evaluated")


if __name__ == "__main__":
    main()
