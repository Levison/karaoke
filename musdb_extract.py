#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
musdb_extract.py — copy MUSDB18 stereo mixes into test_songs/ for karaoke_lrc / eval_lrc.

Word-level ground truth from musdb_truth.py only matches the official Zenodo stems.
Do not substitute YouTube rips for these names.

Request MUSDB18 (academic use): https://sigsep.github.io/datasets/musdb.html
Unzip so you have .../test/Timboz - Pony.stem.mp4 (and train/ if needed).

Usage:
    export MUSDB18_PATH=/path/to/musdb18
    uv run musdb_extract.py                         # default: 4 Heavy Metal test songs
    uv run musdb_extract.py "Timboz - Pony"
    uv run musdb_extract.py --root D:/datasets/musdb18 "Hollow Ground - Ill Fate"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "test_songs"

# Same set as musdb_truth.py METAL — harsh vocals / metal benchmark.
SCREAMY_MUSDB = [
    "Hollow Ground - Ill Fate",
    "James Elder & Mark M Thompson - The English Actor",
    "Timboz - Pony",
    "We Fell From The Sky - Not You",
]


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise SystemExit("ffmpeg not found on PATH")
    return exe


def find_stem(root: Path, name: str) -> Path | None:
    for sub in ("test", "train"):
        path = root / sub / f"{name}.stem.mp4"
        if path.is_file():
            return path
    return None


def extract(stem_path: Path, out_path: Path, dry_run: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg(),
        "-y",
        "-i",
        str(stem_path),
        "-map",
        "0:0",
        "-c",
        "copy",
        str(out_path),
    ]
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit((proc.stderr or proc.stdout).strip() or "ffmpeg failed")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("songs", nargs="*", default=SCREAMY_MUSDB)
    parser.add_argument("--root", type=Path, help="MUSDB18 folder (default: MUSDB18_PATH env)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ext", default="m4a", help="output container (default: m4a)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.root is not None:
        root = args.root
    else:
        env = os.environ.get("MUSDB18_PATH", "").strip()
        if not env:
            raise SystemExit(
                "Set MUSDB18_PATH or pass --root to your unzipped musdb18 folder "
                "(see https://sigsep.github.io/datasets/musdb.html)."
            )
        root = Path(env)
    if not root.is_dir():
        raise SystemExit(f"MUSDB18 folder not found: {root}")

    for name in args.songs:
        stem = find_stem(root, name)
        if stem is None:
            print(f"skip   {name}: no {name}.stem.mp4 under {root}/test or train/", file=sys.stderr)
            continue
        out = args.out_dir / f"{name}.{args.ext.lstrip('.')}"
        print(f"\n{name}")
        extract(stem, out, args.dry_run)
        if not args.dry_run:
            print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
