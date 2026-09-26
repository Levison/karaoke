#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
musdb_truth.py — fetch word-level lyric timings for MUSDB18 test songs and write
them as eval_lrc.py ground truth.

Two annotation sets, both for the MUSDB18 test set, are merged:
  words  zenodo.org/records/15547046 (MIT): one row per word, "start,word", start
         times set by hand against the isolated vocals. No end times, no lines.
  lines  huggingface.co/datasets/jazasyed/musdb-alt (CC BY-NC-SA 4.0): line text
         with start/end times. Covers 39 of the 45 songs (not Timboz - Pony,
         whose screamed vocals it calls unintelligible).

The two teams transcribed the words differently (72-92% overlap), so line breaks
come from times, not text: each word joins the latest line starting no later than
0.1 s after it, and a word starting over 0.5 s after that line's end starts a new
line (ad-libs the line set left out). On the three metal songs that have both,
this matches text alignment on 96 of 105 line ends; nearly all the rest are words
text alignment couldn't pair up, which start within 0.2 s of a line start. Songs
without line times get no breaks, and eval_lrc.py then skips the break metrics.

word_end isn't annotated: it's the next word's start, capped at 1 s. eval_lrc.py
only reads word_start and line_end.

Audio isn't included. MUSDB18 is on Zenodo behind an access request (academic
use only): https://sigsep.github.io/datasets/musdb.html. Each song is a
"<name>.stem.mp4"; stream 0 is the mix, stream 4 the vocals. Extract the mix
without re-encoding so the file stem matches the ground truth:
    ffmpeg -i "Timboz - Pony.stem.mp4" -map 0:0 -c copy "test_songs/Timboz - Pony.m4a"

Usage:
    uv run musdb_truth.py                         # the 4 Heavy Metal songs
    uv run musdb_truth.py "Zeno - Signs" ...      # any MUSDB18 test songs
    uv run musdb_truth.py --out some/dir          # default: test_songs/groundtruth
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

WORDS_URL = "https://zenodo.org/api/records/15547046/files/{}_align.csv/content"
LINES_URL = "https://huggingface.co/datasets/jazasyed/musdb-alt/resolve/main/data/test.jsonl"

# MUSDB18 test songs tagged Heavy Metal in sigsep's tracklist.csv.
METAL = [
    "Hollow Ground - Ill Fate",
    "James Elder & Mark M Thompson - The English Actor",
    "Timboz - Pony",
    "We Fell From The Sky - Not You",
]

START_SLACK = 0.1   # a word may start this much before its line's annotated start
LATE_AFTER = 0.5    # a word starting this long after its line's end begins a new line
MAX_WORD_LEN = 1.0  # cap on the inferred word_end


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def read_words(name: str) -> list[tuple[float, str]]:
    text = fetch(WORDS_URL.format(urllib.parse.quote(name)))
    words = [(float(start), word.strip()) for start, word in csv.reader(text.splitlines()) if word.strip()]
    if any(len(w.split()) != 1 for _, w in words):
        raise ValueError(f"{name}: a word contains whitespace")
    return sorted(words)


def line_ends(starts: list[float], lines: list[dict]) -> list[bool]:
    """True for the last word of each lyric line (see module docstring)."""
    line_starts = [ln["start"] for ln in lines]
    groups = []
    for s in starts:
        i = bisect.bisect_right(line_starts, s + START_SLACK) - 1
        late = i >= 0 and s > lines[i]["end"] + LATE_AFTER
        groups.append((i, late))
    return [a != b for a, b in zip(groups, groups[1:])] + [True]


def write_truth(name: str, words: list[tuple[float, str]], ends: list[bool] | None, out: Path) -> None:
    starts = [s for s, _ in words]
    word_ends = [min(nxt, s + MAX_WORD_LEN) for s, nxt in zip(starts, starts[1:] + [float("inf")])]
    base = out / name

    base.with_name(name + ".words.txt").write_text("".join(w + "\n" for _, w in words), encoding="utf-8")
    with base.with_name(name + ".words.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word_start", "word_end", "line_end"])
        for k, (s, e) in enumerate(zip(starts, word_ends)):
            writer.writerow([f"{s:.6f}", f"{e:.6f}", f"{e:.6f}" if ends and ends[k] else "nan"])

    # Lyrics as lines, the way JamendoLyrics' <stem>.txt holds them; one line if unknown.
    text = "".join(w + ("\n" if ends and ends[k] else " ") for k, (_, w) in enumerate(words))
    base.with_name(name + ".txt").write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("songs", nargs="*", default=METAL, help="MUSDB18 test song names (default: the metal ones)")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "test_songs" / "groundtruth")
    args = parser.parse_args()

    all_lines = {row["name"]: row["lines"] for row in map(json.loads, fetch(LINES_URL).splitlines())}
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.songs:
        words = read_words(name)
        lines = all_lines.get(name)
        ends = line_ends([s for s, _ in words], lines) if lines else None
        write_truth(name, words, ends, args.out)
        breaks = f"{sum(ends)} lines" if ends else "no line times"
        print(f"{name}: {len(words)} words, {breaks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
