#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["mutagen>=1.47"]
# ///
"""
fetch_lyrics_dir.py — fetch LRCLIB/Jamendo lyrics sidecars for every audio file in a folder.

    uv run fetch_lyrics_dir.py test_songs/chris_karaoke
    uv run fetch_lyrics_dir.py test_songs/chris_karaoke --force   # re-fetch even if .lyrics.txt exists
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mutagen

from fix_lrc_tags import read_tags
from known_lyrics import find_lyrics

AUDIO_EXTS = {".mp3", ".flac", ".m4a", ".ogg", ".opus", ".wav"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dir", type=Path, help="folder of audio files")
    parser.add_argument("--force", action="store_true", help="overwrite existing .lyrics.txt")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    if not args.dir.is_dir():
        print(f"Not a directory: {args.dir}", file=sys.stderr)
        return 2

    ok = miss = skip = 0
    for audio in sorted(p for p in args.dir.iterdir() if p.suffix.lower() in AUDIO_EXTS):
        sidecar = audio.with_name(audio.stem + ".lyrics.txt")
        if sidecar.is_file() and not args.force:
            skip += 1
            continue
        if args.force and sidecar.is_file():
            sidecar.unlink()
        title, artist = read_tags(audio)
        info = mutagen.File(audio)
        duration = float(info.info.length) if info and info.info else 0.0
        found = find_lyrics(audio, artist, title, duration, None, True)
        if found:
            ok += 1
            print(f"ok   {audio.name}")
        else:
            miss += 1
            print(f"miss {audio.name}  ({artist} / {title}, {duration:.0f}s)")
    print(f"\n{ok} fetched, {miss} missed, {skip} skipped (already had sidecars)")
    return 0 if miss == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
