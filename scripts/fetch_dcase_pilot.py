"""Fetch a fixed DCASE2019 synthetic pilot via ZIP byte ranges, without 1.19 GB download."""

import argparse
import csv
import hashlib
import io
import json
import random
import struct
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


URL = "https://zenodo.org/records/2583796/files/Synthetic_dataset.zip"
ARCHIVE_SIZE = 1190326055


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/dcase2019"))
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
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
    selected = sorted(random.Random(args.seed).sample(audio, args.count), key=lambda i: i.filename)
    annotation = next(i for i in infos if i.filename == "synthetic_dataset.csv")
    annotation_path = args.output / "synthetic_dataset.csv"
    if not annotation_path.exists() or zlib.crc32(annotation_path.read_bytes()) != annotation.CRC:
        annotation_path.write_bytes(fetch_member(annotation, directory_start))
    wav_dir = args.output / "wav"
    wav_dir.mkdir(exist_ok=True)

    def download(info):
        path = wav_dir / Path(info.filename).name
        if not path.exists() or zlib.crc32(path.read_bytes()) != info.CRC:
            path.write_bytes(fetch_member(info, directory_start))
        return path.name, info.filename, f"{info.CRC:08x}", hashlib.sha256(path.read_bytes()).hexdigest()

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = []
        for row in pool.map(download, selected):
            rows.append(row)
            if len(rows) % 8 == 0:
                print(f"Verified {len(rows)}/{len(selected)} recordings", flush=True)
    with (args.output / "pilot_manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filename", "source_member", "zip_crc32", "sha256"))
        writer.writerows(rows)
    (args.output / "pilot_source.json").write_text(json.dumps({
        "source": URL, "archive_bytes": ARCHIVE_SIZE, "archive_audio_count": len(audio),
        "selection": "random.Random(seed).sample(sorted ZIP audio names, count)",
        "seed": args.seed, "count": args.count, "original_validation_split": False,
        "integrity": "ZIP directory plus individual member size/CRC32; full archive MD5 not checked",
        "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    print(f"Saved fixed pilot to {args.output}; no training/test split reconstructed")


if __name__ == "__main__":
    main()
