"""Download the real DCASE2019 validation set (1,168 clips) from YouTube.

    python -m pip install -r requirements.txt     # yt-dlp; also needs ffmpeg
    python scripts/fetch_validation.py                      # resumable; rerun to retry

This is the official DESED procedure: each clip is cut from its YouTube video at
the official start/end times. Videos that were removed or made private cannot be
fetched; they are listed in missing.tsv (the organizers supply such files on
request). Re-downloaded audio is not guaranteed to be byte-identical to the
organizers' copy. Writes data/raw/dcase2019_validation_youtube/{wav/, log.jsonl,
manifest.json, missing.tsv}. Labels: the official validation.tsv in the pinned
2019 upstream checkout (cache/DCASE2019_task4).
"""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "cache/DCASE2019_task4/dataset/metadata/validation/validation.tsv"
OUT = ROOT / "data/raw/dcase2019_validation_youtube"
LOCK = threading.Lock()


def clip_info(filename):
    """Y<11-char video id>_<start>_<end>.wav -> (video id, start s, end s)."""
    video, start, end = Path(filename).stem.rsplit("_", 2)
    return video[1:], float(start), float(end)


def duration_s(path):
    with wave.open(str(path)) as audio:
        return audio.getnframes() / audio.getframerate()


def fetch(filename, extra_args, retries):
    target = OUT / "wav" / filename
    video, start, end = clip_info(filename)
    if target.exists():
        return {"filename": filename, "status": "ok", "cached": True}
    error = ""
    for attempt in range(retries + 1):
        with tempfile.TemporaryDirectory() as tmp:
            command = [sys.executable, "-m", "yt_dlp", "--quiet", "--no-warnings",
                       "--no-playlist", "-f", "bestaudio/best",
                       "--download-sections", f"*{start}-{end}", "--force-keyframes-at-cuts",
                       "-o", f"{tmp}/clip.%(ext)s", *extra_args,
                       f"https://www.youtube.com/watch?v={video}"]
            result = subprocess.run(command, capture_output=True, text=True)
            downloaded = [p for p in Path(tmp).glob("clip.*")]
            if result.returncode == 0 and downloaded:
                # Official clips are 44.1 kHz PCM16; keep native channels and cap the
                # length at the official segment duration.
                partial = target.with_suffix(".partial.wav")
                convert = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(downloaded[0]),
                           "-t", f"{end - start:.3f}", "-ar", "44100", "-c:a", "pcm_s16le", str(partial)]
                converted = subprocess.run(convert, capture_output=True, text=True)
                if converted.returncode == 0:
                    partial.replace(target)
                    return {"filename": filename, "status": "ok", "video": video,
                            "start_s": start, "end_s": end, "duration_s": duration_s(target),
                            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
                error = converted.stderr.strip()[-300:]
            else:
                error = (result.stderr or "no file produced").strip()[-300:]
        if "unavailable" in error.lower() or "private" in error.lower() or "removed" in error.lower():
            break  # permanent; retrying will not help
        time.sleep(2 * (attempt + 1))
    return {"filename": filename, "status": "missing", "video": video, "error": error}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=3, help="parallel downloads (keep small)")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--cookies-from-browser", help="e.g. chrome, if YouTube asks to sign in")
    args = parser.parse_args()
    for tool in ("ffmpeg",):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found; install it first (macOS: brew install ffmpeg)")
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise SystemExit("yt-dlp not installed: python -m pip install -r requirements.txt")
    extra = ["--cookies-from-browser", args.cookies_from_browser] if args.cookies_from_browser else []

    with LABELS.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    names = sorted({row["filename"] for row in rows})
    assert len(names) == 1168, len(names)
    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(LABELS, OUT / "validation.tsv")
    log = (OUT / "log.jsonl").open("a")
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(lambda n: fetch(n, extra, args.retries), names):
            results.append(record)
            with LOCK:
                log.write(json.dumps(record) + "\n")
                log.flush()
            done += 1
            if done % 25 == 0 or done == len(names):
                ok = sum(r["status"] == "ok" for r in results)
                print(f"{done}/{len(names)} processed, {ok} available", flush=True)
    log.close()

    ok = [r for r in results if r["status"] == "ok"]
    files = []
    for record in ok:
        path = OUT / "wav" / record["filename"]
        files.append({"filename": record["filename"], "duration_s": duration_s(path),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    missing = [r for r in results if r["status"] != "ok"]
    manifest = {"labels": "official DCASE2019 validation.tsv (upstream 90d35b6)",
                "official_recordings": len(names), "available": len(files), "missing": len(missing),
                "source": "YouTube, cut at official start/end times; not byte-identical to the organizers' copy",
                "files": files}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (OUT / "missing.tsv").open("w") as handle:
        handle.write("filename\terror\n")
        for record in missing:
            handle.write(f"{record['filename']}\t{record.get('error', '').replace(chr(9), ' ')[:200]}\n")
    short = sum(f["duration_s"] < 9.0 for f in files)
    print(f"Done: {len(files)} of {len(names)} clips available ({len(files) / len(names):.0%}); "
          f"{len(missing)} missing (see missing.tsv); {short} clips shorter than 9 s")


if __name__ == "__main__":
    main()
