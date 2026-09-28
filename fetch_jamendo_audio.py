#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["mutagen>=1.47"]
# ///
"""
fetch_jamendo_audio.py — download Jamendo tracks by ID into test_songs/.

Jamendo CC songs only. IDs come from your own list (copy fetch_jamendo_audio.example.json).

    cp fetch_jamendo_audio.example.json fetch_jamendo_audio.json
    uv run fetch_jamendo_audio.py
    uv run fetch_jamendo_audio.py --file my_jamendo.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import mutagen

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "test_songs"
DEFAULT_LIST = ROOT / "fetch_jamendo_audio.json"
DOWNLOAD = "https://prod-1.storage.jamendo.com/download/track/{id}/mp32/"


def load_list(path: Path) -> tuple[Path, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = Path(data.get("out_dir", "test_songs"))
    if not out.is_absolute():
        out = (path.parent / out).resolve()
    rows = data.get("tracks")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f'{path}: need a non-empty "tracks" array')
    return out, rows


def download_track(track_id: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = DOWNLOAD.format(id=track_id)
    request = urllib.request.Request(url, headers={"User-Agent": "karaoke_lrc"})
    with urllib.request.urlopen(request, timeout=120) as response:
        dest.write_bytes(response.read())


def tag(path: Path, artist: str, title: str) -> None:
    audio = mutagen.File(path, easy=True)
    if audio is None:
        raise SystemExit(f"Could not tag {path}")
    if artist:
        audio["artist"] = artist
    if title:
        audio["title"] = title
    audio.save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-f", "--file", type=Path, default=DEFAULT_LIST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    if not args.file.is_file():
        example = ROOT / "fetch_jamendo_audio.example.json"
        raise SystemExit(f"No list at {args.file}. Copy {example.name} and add track ids.")

    out_dir, tracks = load_list(args.file)
    for i, row in enumerate(tracks, start=1):
        tid = row.get("id")
        stem = row.get("stem")
        if tid is None or not stem:
            raise SystemExit(f"tracks[{i - 1}] needs id and stem")
        dest = out_dir / f"{stem}.mp3"
        print(f"\n{row.get('artist', '?')} — {row.get('title', stem)}  (id {tid})")
        if args.dry_run:
            print(f"  -> would write {dest}")
            continue
        download_track(int(tid), dest)
        tag(dest, str(row.get("artist") or ""), str(row.get("title") or stem))
        print(f"  -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
