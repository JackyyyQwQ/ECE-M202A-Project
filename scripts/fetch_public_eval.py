"""Download and unpack the official DCASE2019 Task 4 public evaluation set.

    python scripts/fetch_public_eval.py

Fetches Zenodo record 4560759 (DESED_public_eval.tar.gz, 890 MB) in resumable
16 MB ranges, checks the published MD5, then writes data/raw/dcase2019_public_eval/
{wav/*.wav, public.tsv, ..., manifest.json} with a SHA-256 per recording.
"""

import hashlib
import json
import shutil
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data/raw/dcase2019_public_eval"
URL = "https://zenodo.org/records/4560759/files/DESED_public_eval.tar.gz"
SIZE = 889_698_336
MD5 = "c1bf328257523f7e93244f3c1b160eb4"
BLOCK = 16 * 1024 * 1024


def md5(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "md5").hexdigest()


def fetch_block(parts, start, end):
    path = parts / f"{start:012d}.part"
    if path.exists() and path.stat().st_size == end - start + 1:
        return path
    for attempt in range(3):
        try:
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=90) as response:
                assert response.status == 206
                body = response.read(end - start + 2)
            assert len(body) == end - start + 1
            temp = path.with_suffix(".tmp")
            temp.write_bytes(body)
            temp.replace(path)
            return path
        except (OSError, AssertionError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def download(archive):
    parts = FOLDER / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    bounds = [(s, min(s + BLOCK, SIZE) - 1) for s in range(0, SIZE, BLOCK)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        paths = list(pool.map(lambda b: fetch_block(parts, *b), bounds))
    temp = archive.with_suffix(".download")
    with temp.open("wb") as target:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, target)
    if md5(temp) != MD5:
        raise SystemExit("Archive MD5 mismatch; delete data/raw/dcase2019_public_eval/parts and retry")
    temp.replace(archive)
    shutil.rmtree(parts)


def main():
    archive = FOLDER / "DESED_public_eval.tar.gz"
    if not archive.exists() or md5(archive) != MD5:
        download(archive)
    print("Archive MD5 verified:", MD5)
    (FOLDER / "wav").mkdir(parents=True, exist_ok=True)
    members = []
    with tarfile.open(archive, "r:gz") as tar:
        for item in tar:
            name = Path(item.name).name
            if not item.isfile() or name.startswith("._"):
                continue
            with tar.extractfile(item) as source:
                body = source.read()
            if name.endswith(".wav"):
                (FOLDER / "wav" / name).write_bytes(body)
                members.append({"filename": name, "source_member": item.name,
                                "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)})
            elif name.endswith((".tsv", ".csv", ".txt", ".md")):
                (FOLDER / name).write_bytes(body)
    assert len(members) == 692, len(members)
    manifest = {"record": 4560759, "source": URL, "archive_bytes": SIZE, "archive_md5": MD5,
                "files": sorted(members, key=lambda r: r["filename"])}
    (FOLDER / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Extracted and hashed 692 recordings into", FOLDER)


if __name__ == "__main__":
    main()
