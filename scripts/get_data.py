"""Download and extract the RAVDESS speech corpus.

    python scripts/get_data.py

The corpus is not committed: it is 200 MB and its licence (CC BY-NC-SA 4.0)
does not allow redistribution. This script reproduces data/ravdess/ from the
official Zenodo record.
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402

URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
ZIP_PATH = C.DATA_DIR / "ravdess.zip"
EXPECTED_WAVS = 1440


def report(block_num: int, block_size: int, total: int) -> None:
    done = block_num * block_size
    pct = min(100.0, done * 100.0 / total) if total > 0 else 0.0
    print(f"\r  {pct:5.1f} %  ({done / 1e6:.0f} / {total / 1e6:.0f} MB)", end="", flush=True)


def main() -> int:
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    n_existing = len(list(C.RAVDESS_DIR.rglob("*.wav"))) if C.RAVDESS_DIR.exists() else 0
    if n_existing == EXPECTED_WAVS:
        print(f"already present: {n_existing} wav files in {C.RAVDESS_DIR}")
        return 0

    if not ZIP_PATH.exists():
        print(f"downloading RAVDESS speech audio from Zenodo ...")
        urllib.request.urlretrieve(URL, ZIP_PATH, reporthook=report)
        print()

    print(f"extracting to {C.RAVDESS_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(C.RAVDESS_DIR)

    n = len(list(C.RAVDESS_DIR.rglob("*.wav")))
    print(f"{n} wav files, {len(list(C.RAVDESS_DIR.glob('Actor_*')))} actors")
    if n != EXPECTED_WAVS:
        print(f"WARNING: expected {EXPECTED_WAVS} files, found {n}")
        return 1

    ZIP_PATH.unlink()          # 200 MB we no longer need
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
