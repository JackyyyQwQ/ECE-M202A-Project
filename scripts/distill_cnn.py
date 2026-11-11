"""Distil a compact depthwise-separable CNN (DS-CNN) from the 2023 reference CNN.

The student reproduces the reference CNN's 128-D features (one per 64 ms frame);
the official BiGRU + heads stay frozen and unchanged. No labels are used for training.

    python scripts/distill_cnn.py cache                       # teacher targets (once)
    python scripts/distill_cnn.py train --run ds_mse          # feature MSE only
    python scripts/distill_cnn.py train --run ds_out --output-weight 1
    python scripts/distill_cnn.py eval --run ds_mse           # Event F1 on validation

Committed students (no retraining needed): models/distill/{ds_mse, ds_w025}/. `eval`
uses output/distill/<run>/student.pt when present, otherwise the committed copy.

Training audio: DCASE2019 synthetic (data/raw/dcase2019, scripts/fetch_dcase_pilot.py --all).
Evaluation: validation (development) set only; never the public evaluation set.
Same frontend as the teacher, incl. per-clip [0, 1] normalisation, which an MCU
cannot do online; this prototype isolates the distillation error only.
Writes cache/distill/*.npz and output/distill/<run>/.
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from lookahead_sweep import POSTPROCESSING, ROOT, score  # also puts ROOT on sys.path
from scripts import pretrained_sed
from scripts.pretrained_sed import CLASSES

RATE = 16000
CLIP = 10 * RATE                      # training clips padded/trimmed to 10 s
SYNTH = ROOT / "data/raw/dcase2019"
VALID = ROOT / "data/raw/dcase2019_validation_youtube"
CACHE = ROOT / "cache/distill"
OUT = ROOT / "output/distill"
RELEASED = ROOT / "models/distill"    # committed students: models/distill/<run>/{student.pt, train.json}


# ---------------------------------------------------------------- teacher side
def load_teacher():
    model, mel, _ = pretrained_sed.load_detector(ROOT / "cache/DESED_task", pretrained_sed.CHECKPOINT)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, mel


def cnn_input(mel_batch):
    """[B, 128 mel, T] -> [B, 1, T, 128], as CRNN.forward does."""
    return mel_batch.transpose(1, 2).unsqueeze(1)


def teacher_features(model, mel_batch):
    x = model.cnn(cnn_input(mel_batch))          # [B, 128, T/4, 1]
    return x.squeeze(-1).permute(0, 2, 1)        # [B, T/4, 128]


def head(model, feats):
    """Frozen BiGRU + dense + sigmoid: [B, T, 128] -> frame probabilities [B, T, classes]."""
    strong, _ = model._get_logits(model.rnn(feats), None)
    if strong.shape[-1] != len(CLASSES):
        strong = strong.transpose(1, 2)
    return strong


def read_wave(path, resamplers):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.mean(axis=1))
    if rate != RATE:
        wave = resamplers.setdefault(rate, torchaudio.transforms.Resample(rate, RATE))(wave)
    return wave, len(audio) / rate


# ---------------------------------------------------------------- student
class DSBlock(nn.Sequential):
    """Depthwise 3x3 -> BN -> ReLU -> (pool) -> pointwise 1x1 -> BN -> ReLU."""
    def __init__(self, cin, cout, pool):
        super().__init__(
            nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(),
            nn.AvgPool2d(pool) if pool != (1, 1) else nn.Identity(),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


class Student(nn.Module):
    """[B, 1, T, 128 mel] -> [B, T/4, 128]. Time pooling (2, 2) matches the teacher exactly."""
    def __init__(self, width=1.0):
        super().__init__()
        w0, w1, w2 = (max(8, int(c * width)) for c in (32, 64, 128))
        self.stem = nn.Sequential(nn.Conv2d(1, w0, 3, padding=1, stride=(1, 2), bias=False),
                                  nn.BatchNorm2d(w0), nn.ReLU())                  # freq 128 -> 64
        self.blocks = nn.Sequential(DSBlock(w0, w1, (2, 2)), DSBlock(w1, w1, (2, 2)),  # T/4, freq 16
                                    DSBlock(w1, w2, (1, 2)), DSBlock(w2, w2, (1, 2)),  # freq 4
                                    DSBlock(w2, w2, (1, 1)))
        self.project = nn.Conv1d(w2, 128, 1)                                     # linear, no activation

    def forward(self, x):
        x = self.blocks(self.stem(x)).mean(dim=3)                                # [B, w2, T/4]
        return self.project(x).transpose(1, 2)


def macs_per_second(module, x):
    """Multiply-accumulates of all conv layers for input x, scaled to one second of audio."""
    total = [0]
    def hook(layer, _, out):
        k = int(np.prod(layer.kernel_size))
        total[0] += out.numel() * (layer.in_channels // layer.groups) * k
    handles = [m.register_forward_hook(hook) for m in module.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d))]
    with torch.inference_mode():
        module(x)
    for h in handles:
        h.remove()
    seconds = x.shape[2] * 256 / RATE
    return total[0] / seconds


# ---------------------------------------------------------------- data
def annotations():
    events = defaultdict(list)
    with (VALID / "validation.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            events[row["filename"]]
            if row["event_label"]:
                events[row["filename"]].append((row["event_label"], float(row["onset"]), float(row["offset"])))
    return events


def cache(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    model, mel = load_teacher()
    resamplers = {}
    with torch.inference_mode():
        # Training audio: fixed 10 s clips.
        names = sorted(p.name for p in (SYNTH / "wav").glob("*.wav"))
        mels, feats, probs = [], [], []
        for i in range(0, len(names), args.batch):
            waves = []
            for n in names[i:i + args.batch]:
                w, _ = read_wave(SYNTH / "wav" / n, resamplers)
                waves.append(F.pad(w, (0, max(0, CLIP - len(w))))[:CLIP])
            m = pretrained_sed.normalized_mel(torch.stack(waves), mel)
            f = teacher_features(model, m)
            mels.append(m.half()); feats.append(f.half()); probs.append(head(model, f).half())
            print(f"synthetic {min(i + args.batch, len(names))}/{len(names)}", flush=True)
        np.savez(CACHE / "synthetic.npz", names=np.array(names), mel=torch.cat(mels).numpy(),
                 feats=torch.cat(feats).numpy(), probs=torch.cat(probs).numpy())

        # Validation audio: full available length, exactly as the lookahead "full" run.
        names = sorted(n for n in annotations() if (VALID / "wav" / n).exists())
        rows, check = [], None
        for j, n in enumerate(names):
            w, duration = read_wave(VALID / "wav" / n, resamplers)
            m = pretrained_sed.normalized_mel(w[None], mel)
            f = teacher_features(model, m)
            if check is None:  # the split teacher must equal the official forward pass
                strong, _ = model(m)
                check = float((head(model, f)[0] - strong[0].T).abs().max())
                assert check < 1e-5, f"split CNN/BiGRU differs from CRNN.forward by {check}"
            rows.append((n, m[0].half().numpy(), f[0].half().numpy(), len(w), duration))
            if (j + 1) % 100 == 0 or j + 1 == len(names):
                print(f"validation {j + 1}/{len(names)}", flush=True)
        T = max(r[1].shape[1] for r in rows)
        mel_arr = np.zeros((len(rows), 128, T), np.float16)
        for k, r in enumerate(rows):
            mel_arr[k, :, :r[1].shape[1]] = r[1]
        np.savez(CACHE / "validation.npz", names=np.array([r[0] for r in rows]), mel=mel_arr,
                 mel_frames=np.array([r[1].shape[1] for r in rows]),
                 usable=np.array([r[3] for r in rows]), durations=np.array([r[4] for r in rows]))
    print(f"Saved {CACHE}; split-teacher check max |diff| = {check:.2e}")


# ---------------------------------------------------------------- train
def pick_device(name):
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def train(args):
    device = pick_device(args.device)
    data = np.load(CACHE / "synthetic.npz")
    mel = torch.from_numpy(data["mel"])
    feats = torch.from_numpy(data["feats"]).float()
    probs = torch.from_numpy(data["probs"]).float()
    order = torch.randperm(len(mel), generator=torch.Generator().manual_seed(args.seed))
    hold = order[: max(1, len(mel) // 20)]       # 5 % synthetic hold-out for the training curve
    fit = order[len(hold):]
    mean = feats[fit].mean(dim=(0, 1)); std = feats[fit].std(dim=(0, 1)) + 1e-3

    model, _ = load_teacher()
    model.to(device)
    student = Student(args.width).to(device)
    params = sum(p.numel() for p in student.parameters())
    student_macs = macs_per_second(student.cpu(), cnn_input(mel[:1].float())); student.to(device)
    teacher_macs = macs_per_second(model.cnn.cpu(), cnn_input(mel[:1].float())); model.to(device)
    print(f"student: {params:,} parameters, {student_macs / 1e6:.1f} M MAC/s "
          f"(teacher CNN {teacher_macs / 1e6:.1f} M MAC/s); device {device}", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(fit) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    mean, std = mean.to(device), std.to(device)

    def losses(idx):
        m = mel[idx].float().to(device)
        t, p = feats[idx].to(device), probs[idx].to(device)
        s = student(cnn_input(m))
        feature = F.mse_loss((s - mean) / std, (t - mean) / std)
        output = F.binary_cross_entropy(head(model, s).clamp(1e-6, 1 - 1e-6), p) if args.output_weight else s.new_zeros(())
        return feature, output

    run = OUT / args.run
    run.mkdir(parents=True, exist_ok=True)
    log, best = [], float("inf")
    for epoch in range(args.epochs):
        start = time.time()
        student.train()
        perm = fit[torch.randperm(len(fit))]
        for i in range(0, len(perm), args.batch):
            feature, output = losses(perm[i:i + args.batch])
            loss = feature + args.output_weight * output
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        student.eval()
        with torch.no_grad():
            parts = [losses(hold[i:i + args.batch]) for i in range(0, len(hold), args.batch)]
        hf = float(np.mean([float(a) for a, _ in parts])); ho = float(np.mean([float(b) for _, b in parts]))
        log.append({"epoch": epoch + 1, "holdout_feature_mse": hf, "holdout_output_bce": ho,
                    "seconds": time.time() - start})
        print(f"epoch {epoch + 1:3d}  hold-out feature MSE {hf:.4f}  output BCE {ho:.4f}  "
              f"({time.time() - start:.0f} s)", flush=True)
        total = hf + args.output_weight * ho
        if total < best:
            best = total
            torch.save(student.state_dict(), run / "student.pt")
    (run / "train.json").write_text(json.dumps({
        "args": vars(args), "parameters": params, "student_mac_per_s": student_macs,
        "teacher_cnn_mac_per_s": teacher_macs, "training_clips": len(fit), "holdout_clips": len(hold),
        "log": log}, indent=2) + "\n")
    print(f"Saved {run / 'student.pt'}")


# ---------------------------------------------------------------- evaluate
def evaluate(args):
    run = OUT / args.run
    source = run if (run / "student.pt").exists() else RELEASED / args.run   # trained here, or committed
    config = json.loads((source / "train.json").read_text())["args"]
    run.mkdir(parents=True, exist_ok=True)
    model, _ = load_teacher()
    student = Student(config["width"])
    student.load_state_dict(torch.load(source / "student.pt", map_location="cpu", weights_only=True))
    print(f"student weights: {source.relative_to(ROOT) / 'student.pt'}")
    student.eval()
    data = dict(np.load(CACHE / "validation.npz"))   # read each npz array once
    names = [str(n) for n in data["names"]]
    usable = dict(zip(names, data["usable"].tolist()))
    durations = dict(zip(names, data["durations"].tolist()))
    events = annotations()
    outputs = {"teacher": {}, "student": {}}
    feats = {"teacher": [], "student": []}
    with torch.inference_mode():
        for k, n in enumerate(names):
            m = torch.from_numpy(data["mel"][k, :, :data["mel_frames"][k]]).float()[None]
            for who, f in (("teacher", teacher_features(model, m)), ("student", student(cnn_input(m)))):
                outputs[who][n] = head(model, f)[0].numpy()
                feats[who].append(f[0].half().numpy())
    frames = np.array([len(f) for f in feats["teacher"]])
    np.savez_compressed(run / "validation_features.npz", names=np.array(names), frames=frames,
                        **{f"{who}_feats": np.concatenate(v) for who, v in feats.items()},
                        **{f"{who}_scores": np.concatenate([outputs[who][n] for n in names]) for who in outputs})
    results = {}
    for who in outputs:
        for rule, decide in POSTPROCESSING.items():
            results[f"{who} | {rule}"] = score(names, outputs[who], events, durations, usable, decide)
    (run / "eval.json").write_text(json.dumps({"recordings": len(names), "results": results},
                                              indent=2, default=float) + "\n")
    print(f"validation: {len(names)} recordings, full-clip context\n"
          f"{'features | post-processing':<42} {'frame F1':>9} {'event uF1':>10} {'event F1':>9} {'events':>7}")
    for key, r in results.items():
        print(f"{key:<42} {r['frame_micro']['f1']:9.4f} {r['event_micro_f1']:10.4f} "
              f"{r['event_macro_f1']:9.4f} {r['predicted_events']:7d}")
    print(f"Saved {run / 'eval.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cache"); c.add_argument("--batch", type=int, default=16)
    t = sub.add_parser("train")
    t.add_argument("--run", required=True)
    t.add_argument("--width", type=float, default=1.0, help="channel multiplier (1.0 = 32/64/128)")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--output-weight", type=float, default=0.0,
                   help="weight of BCE between frozen-BiGRU outputs on student vs teacher features")
    t.add_argument("--device", default="auto", help="auto | mps | cpu")
    t.add_argument("--seed", type=int, default=2026)
    e = sub.add_parser("eval"); e.add_argument("--run", required=True)
    args = parser.parse_args()
    {"cache": cache, "train": train, "eval": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
