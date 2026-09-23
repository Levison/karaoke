#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["mutagen>=1.47"]
# ///
"""
fix_lrc_tags.py — make .lrc sidecars attach to their songs when imported into OpenKara.

OpenKara copies imported audio into its library as media/<sha256>.<ext>, so it can't
match a .lrc by file name. It falls back to the .lrc's [ar:] and [ti:] tags, which must
both equal (ignoring case) the artist and title tags of the audio file
(src-tauri/src/commands/lyrics.rs, import_lyrics_files). karaoke_lrc.py now writes
these itself via read_tags() below; this script fixes older or hand-made .lrc files.

This rewrites just those two header lines from the neighbouring audio file's tags,
the way OpenKara reads them: title falls back to the file name, a missing artist
becomes an empty [ar:]. Timing lines are left untouched. Safe to run repeatedly.

Usage:
    uv run fix_lrc_tags.py                    # every .lrc in test_songs/
    uv run fix_lrc_tags.py some/folder x.lrc  # folders and/or .lrc files
    uv run fix_lrc_tags.py --dry-run          # show what would change
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import mutagen

AUDIO_EXTS = (".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac", ".wma", ".opus", ".aiff", ".aif")
HEADER_TAG = re.compile(r"^\s*\[(ti|ar):[^\n]*$", re.IGNORECASE)


def find_audio(lrc: Path) -> Path | None:
    for ext in AUDIO_EXTS:
        for candidate in (lrc.with_suffix(ext), lrc.with_suffix(ext.upper())):
            if candidate.is_file():
                return candidate
    return None


def read_tags(audio: Path) -> tuple[str, str]:
    """(title, artist) as OpenKara's import sees them."""
    tags = mutagen.File(audio, easy=True)
    values = dict(tags.tags or {}) if tags is not None else {}

    def first(key: str) -> str | None:
        vals = values.get(key) or []
        return str(vals[0]) if vals else None

    return first("title") or audio.stem, first("artist") or ""


def fix(lrc: Path, dry_run: bool) -> str:
    audio = find_audio(lrc)
    if audio is None:
        return f"skip   {lrc.name}: no audio file with the same name next to it"
    title, artist = read_tags(audio)

    raw = lrc.read_bytes()
    text = raw.decode("utf-8-sig")
    newline = "\r\n" if "\r\n" in text else "\n"
    kept = [line for line in text.splitlines() if not HEADER_TAG.match(line)]
    fixed = newline.join([f"[ti:{title}]", f"[ar:{artist}]", *kept]) + newline

    warning = ""
    if "]" in title or "]" in artist:
        # OpenKara's tag parser stops at the first "]", so this song can't auto-match.
        warning = "  <- has ']' in its title/artist: use Edit lyrics in OpenKara for this one"

    if fixed.encode("utf-8") == raw:
        return f"ok     {lrc.name}{warning}"
    if not dry_run:
        lrc.write_text(fixed, encoding="utf-8", newline="")
    verb = "would fix" if dry_run else "fixed "
    return f"{verb} {lrc.name}  ->  [ti:{title}] [ar:{artist}]{warning}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", type=Path, default=[Path(__file__).parent / "test_songs"])
    parser.add_argument("--dry-run", action="store_true", help="print changes without writing")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # e.g. "Cortéz" on a cp1252 console

    lrcs: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            lrcs.extend(sorted(p for p in path.iterdir() if p.suffix.lower() == ".lrc"))
        elif path.suffix.lower() == ".lrc" and path.is_file():
            lrcs.append(path)
        else:
            print(f"skip   {path}: not a folder or .lrc file", file=sys.stderr)
    for lrc in lrcs:
        print(fix(lrc, args.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
