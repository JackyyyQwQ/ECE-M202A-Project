# Pretrained CNN + BiGRU: first offline trial

November 7, 2026. This is an additional baseline candidate, not a deployed encoder
or an online replacement for the proposed causal path. No model was trained.

## Model and input

We use the official DCASE 2023 Task 4 **plain CRNN student**, without BEATs
embeddings. Its CNN, two-layer bidirectional GRU and heads are loaded together
from `legacy_epoch=176-step=20886.ckpt` (1,112,420 parameters). All learned tensors
load strictly; only non-learned `total_ops`/`total_params` profiling counters are
removed. The checkpoint contains student and EMA-teacher weights; this trial
uses the student, matching the plain baseline's reported inference convention.

- [Official recipe](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2023_task4_baseline)
- [Official checkpoint archive](https://zenodo.org/records/7759146)
- Model code revision: `c6bcb45b8b986ccde5c56bb86eefaf9d19b2320c`.
  The upstream checkout remains local/ignored; code is MIT licensed in its LICENSE.
  Checkpoint distribution terms should be checked separately before redistribution.
- Archive MD5: `65e89bb452e141267e41e5c849e05ac8`.
- Selected checkpoint SHA256:
  `16a7f4cef26265ce68623bc275a0dac0cdac1d0aa72c1ace4d89ff6c4d5fc4ce`.

Input: mono 16 kHz; 128 mel bins; 2,048-sample non-periodic Hamming window;
256-sample hop; magnitude mel, amplitude dB clamped to [-50, 80], followed by
per-recording minmax normalization to [-1, 1]. The exact saved window and mel
filter bank are restored. These differ from our existing custom encoder frontend.

For a 10-second input the output is `[1, 10, 156]`, with a **64 ms** timestamp
step (256 samples times temporal subsampling 4). We preserve the official hop-based
grid, ending at 9.984 s; we do not stretch it to 10 seconds. The 10 classes are
alarm, blender, cat, dishes, dog, electric shaver/toothbrush, frying, running water,
speech and vacuum cleaner. Export files retain their official names and order.

## Setup on Mac

From the repository root, activate the project conda environment. These optional
dependencies use a matching torch/torchaudio pair; this trial used macOS 15.8 arm64,
Python 3.11.18. The global/base environment is not changed.

```bash
conda activate "$PWD/.conda"
python -m pip install -r requirements-pretrained.txt
mkdir -p cache checkpoints/dcase2023
git clone --no-checkout https://github.com/DCASE-REPO/DESED_task.git cache/DESED_task
git -C cache/DESED_task checkout c6bcb45b8b986ccde5c56bb86eefaf9d19b2320c
curl -L --fail -o checkpoints/dcase2023/baseline.zip https://zenodo.org/records/7759146/files/dcase2023_task4_baseline.zip
```

Skip clone/download when already present. Verify the archive and extract only
the plain checkpoint:

```bash
python - <<'PY'
import hashlib
import zipfile
from pathlib import Path
folder = Path("checkpoints/dcase2023")
archive = folder / "baseline.zip"
if hashlib.md5(archive.read_bytes()).hexdigest() != "65e89bb452e141267e41e5c849e05ac8":
    raise SystemExit("Checkpoint archive checksum mismatch")
name = "legacy_epoch=176-step=20886.ckpt"
with zipfile.ZipFile(archive) as files:
    (folder / name).write_bytes(files.read("ckpt/" + name))
PY
```

## Run an audio file

```bash
python scripts/pretrained_sed.py /absolute/path/to/audio.wav --output output/my_audio
```

The runner accepts one recording of at most 10 seconds. It averages stereo channels,
resamples to 16 kHz, and zero-pads short recordings to the trained 10-second window.
Longer recordings are rejected explicitly. It writes:

- `scores_raw.tsv`: frame scores with onset/offset timestamps.
- `scores_filtered.tsv`: scores after the default symmetric 7-frame median filter.
- `events.tsv`: events above threshold 0.5; neither threshold nor filter is tuned.
- `run.json`: input/checkpoint hashes, configuration, timing and limitations.

Padding-only frames are omitted and intervals are clipped to real audio duration.
The model and filter still process the padded full window. This is not claimed to
reproduce official benchmark scores, nor to be a calibrated streaming detector.

## Verified smoke run

The public [PANNs example audio](https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/resources/R9_ZSCveAHg_7s.wav)
was downloaded to an ignored local folder. It has no local ground-truth annotations.
To reproduce:

```bash
mkdir -p data/raw/smoke
curl -L --fail -o data/raw/smoke/panns_example.wav https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/resources/R9_ZSCveAHg_7s.wav
python scripts/pretrained_sed.py data/raw/smoke/panns_example.wav --output output/pretrained_smoke --repeats 10
```

The 7-second, 32 kHz recording was resampled and padded to 10 seconds. On this Mac's
CPU with four threads, mel + normalization + model computation took **59.1 ms on
the first call**, then **30.0 ms median** over ten calls (27.4–56.4 ms). File/model
loading, resampling, event decoding, output writing, audio waiting and communication
are excluded. These are local measurements, not a hardware performance guarantee.
See [saved smoke evidence](../results/pretrained_smoke.json).

At the untuned defaults, predictions were alarm [0, 2.368), speech [2.432, 4.032)
and speech [4.800, 6.400) seconds. These are predictions, not verified correct events.
**This smoke run did not evaluate accuracy/F1 or boundary error.** A subsequent
[new 64-recording labeled pilot](pretrained_evaluation.md) measured frame micro F1
39.72%. The original custom-model split manifests are still missing.

## Limits and next experiment

This full-window model uses BiGRU future context, centered STFT, whole-window
normalization and symmetric smoothing. A 30 ms computation does not mean a result
is available 30 ms after an event: input collection and lookahead are separate costs.
Our existing custom 32D features cannot feed this pretrained GRU directly. The
official CNN has not been fitted to the Nano or exported to Zephyr.

The labeled pilot is documented separately; next audit its failure classes and
possible pretraining-data overlap. Then test bounded context at a fixed target time. Test timing, normalization and postprocessing together before claiming online
causality; do not obtain a causal model merely by removing the backward GRU.


## Full-corpus evaluation

The same frozen model and fixed threshold have now been evaluated on all 2,045
DCASE2019 synthetic recordings. Raw frame micro F1 is 41.10% (precision 84.79%,
recall 27.13%); median CPU computation is 28.0 ms per 10-second input. The previous
smoke and pilot results remain unchanged. See the [full report](pretrained_full_report.md)
and [PDF](pretrained_full_report.pdf) for conditions, per-class results and limits.
