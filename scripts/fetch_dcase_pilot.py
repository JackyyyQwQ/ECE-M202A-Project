"""Fetch a fixed DCASE2019 pilot, or the complete archive with --all."""

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import struct
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


URL = "https://zenodo.org/records/2583796/files/Synthetic_dataset.zip"
ARCHIVE_SIZE = 1190326055
ARCHIVE_MD5 = "06ebea7cc467ace800752bd21ec7178d"


def fetch_range(first, last):
    request = urllib.request.Request(URL, headers={"Range": f"bytes={first}-{last}"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
            f"bytes {first}-{last}/"
        ):
            raise RuntimeError("Server did not honor the requested ZIP byte range")
        data = response.read(last - first + 2)
    if len(data) != last - first + 1:
        raise RuntimeError("Incomplete ZIP byte range")
    return data


def fetch_member(info, directory_start):
    offset = info.header_offset + directory_start
    # Local ZIP headers can have up to 65535 bytes of extra metadata.
    last = min(ARCHIVE_SIZE - 1, offset + 30 + len(info.filename.encode()) + 65535 + info.compress_size - 1)
    packet = fetch_range(offset, last)
    if packet[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Invalid local ZIP header: {info.filename}")
    name_size, extra_size = struct.unpack_from("<HH", packet, 26)
    begin = 30 + name_size + extra_size
    compressed = packet[begin:begin + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise ValueError(f"Unsupported ZIP compression: {info.compress_type}")
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise RuntimeError(f"ZIP member size/CRC mismatch: {info.filename}")
    return data


def fetch_archive(folder):
    """Download resumable blocks, then verify the complete official archive."""
    archive = folder / "Synthetic_dataset.zip"
    def checksum(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "md5").hexdigest()
    if archive.exists() and checksum(archive) == ARCHIVE_MD5:
        return archive
    parts = folder / "archive_parts"
    parts.mkdir(exist_ok=True)
    block_size = 16 * 1024 * 1024
    blocks = [(first, min(first + block_size, ARCHIVE_SIZE) - 1)
              for first in range(0, ARCHIVE_SIZE, block_size)]

    def block(bounds):
        first, last = bounds
        path = parts / f"{first:012d}.part"
        if path.exists() and path.stat().st_size == last - first + 1:
            return path
        for attempt in range(3):
            try:
                data = fetch_range(first, last)
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(data)
                temporary.replace(path)
                return path
            except (OSError, RuntimeError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = []
        for path in pool.map(block, blocks):
            paths.append(path)
            print(f"Archive blocks verified: {len(paths)}/{len(blocks)}", flush=True)
    temporary = archive.with_suffix(".download")
    with temporary.open("wb") as destination:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, destination)
    if checksum(temporary) != ARCHIVE_MD5:
        raise RuntimeError("Full archive MD5 mismatch; cached blocks retained for inspection")
    temporary.replace(archive)
    for path in paths:
        path.unlink()
    parts.rmdir()
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/dcase2019"))
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--all", action="store_true", help="Download and evaluate all 2045 recordings")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    directory_path = args.output / "zip_directory.bin"
    if not directory_path.exists():
        directory_path.write_bytes(fetch_range(ARCHIVE_SIZE - 262144, ARCHIVE_SIZE - 1))
    directory = directory_path.read_bytes()
    directory_start = ARCHIVE_SIZE - len(directory)
    with zipfile.ZipFile(io.BytesIO(directory)) as archive:
        infos = archive.infolist()
    audio = sorted((i for i in infos if i.filename.endswith(".wav")), key=lambda i: i.filename)
    if len(audio) != 2045 or not 1 <= args.count <= len(audio):
        raise ValueError("Expected 2045 recordings; count must be within that range")
    selected = audio if args.all else sorted(random.Random(args.seed).sample(audio, args.count), key=lambda i: i.filename)
    archive_path = fetch_archive(args.output) if args.all else None
    annotation = next(i for i in infos if i.filename == "synthetic_dataset.csv")
    annotation_path = args.output / "synthetic_dataset.csv"
    if not annotation_path.exists() or zlib.crc32(annotation_path.read_bytes()) != annotation.CRC:
        annotation_path.write_bytes(fetch_member(annotation, directory_start))
    wav_dir = args.output / "wav"
    wav_dir.mkdir(exist_ok=True)

    def download(info):
        path = wav_dir / Path(info.filename).name
        if not path.exists() or zlib.crc32(path.read_bytes()) != info.CRC:
            if archive_path:
                with zipfile.ZipFile(archive_path) as archive:
                    path.write_bytes(archive.read(info.filename))
            else:
                path.write_bytes(fetch_member(info, directory_start))
        return path.name, info.filename, f"{info.CRC:08x}", hashlib.sha256(path.read_bytes()).hexdigest()

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = []
        for row in pool.map(download, selected):
            rows.append(row)
            if len(rows) % (128 if args.all else 8) == 0:
                print(f"Verified {len(rows)}/{len(selected)} recordings", flush=True)
    prefix = "full" if args.all else "pilot"
    with (args.output / f"{prefix}_manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filename", "source_member", "zip_crc32", "sha256"))
        writer.writerows(rows)
    (args.output / f"{prefix}_source.json").write_text(json.dumps({
        "source": URL, "archive_bytes": ARCHIVE_SIZE, "archive_audio_count": len(audio),
        "selection": "All WAV members, sorted by filename" if args.all else "random.Random(seed).sample(sorted ZIP audio names, count)",
        "seed": None if args.all else args.seed, "count": len(selected), "original_validation_split": False,
        "integrity": f"Complete archive MD5 {ARCHIVE_MD5} and member CRC32" if args.all else "ZIP directory plus individual member size/CRC32; full archive MD5 not checked",
        "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    print(f"Saved {prefix} manifest to {args.output}; no training/test split reconstructed")


if __name__ == "__main__":
    main()
