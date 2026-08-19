"""Fetch a BEIR dataset into ./data.

The dataset is downloaded rather than committed. It is not ours to redistribute,
and a repository that carries its own corpus stops being reviewable at a glance.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def download(name: str, force: bool = False) -> Path:
    target = DATA_DIR / name
    if target.exists() and not force:
        print(f"{target} already exists; use --force to re-download")
        return target

    url = BEIR_URL.format(name=name)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    print(f"  {len(payload) / 1e6:.1f} MB")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(DATA_DIR)
    print(f"Extracted to {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        download(args.dataset, force=args.force)
    except Exception as exc:  # noqa: BLE001 - surface the cause to the user directly
        print(f"Download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
