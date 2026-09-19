#!/usr/bin/env python3
"""
eval_lrc.py — score a generated enhanced .lrc against JamendoLyrics ground truth.

Whisper's words won't match the reference exactly (mishearings, dropped or
extra words), so the two word sequences are aligned first and both timing and
line breaks are measured only on the words that match.

    uv run eval_lrc.py test_songs/Cortez_-_Feel__Stripped_.lrc
    uv run eval_lrc.py test_songs/*.lrc          # skips files with no ground truth

Ground truth is looked up as groundtruth/<stem>.words.txt + .words.csv next to
the .lrc (the layout JamendoLyrics uses: one word per whitespace token, one CSV
row per word with word_start/word_end in seconds, and line_end set on the last
word of each lyric line).

Metrics:
    recall     matched reference words / all reference words
    AAE        mean absolute onset error in seconds, over matched words
               (the standard lyrics-alignment metric)
    median     median onset error
    <0.3s      share of matched words whose onset is within 0.3 s
    brk P/R/F1 line breaks: precision (LRC breaks that are real lyric-line
               ends), recall (lyric-line ends the LRC also breaks at), and F1
    1-word     LRC lines holding a single word (usually a bad break)
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from difflib import SequenceMatcher
from pathlib import Path

WORD_TAG = re.compile(r"<(\d+):(\d+(?:[.:]\d+)?)>([^<]*)")


def norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower()).strip("'")


def parse_lrc(path: Path) -> tuple[list[tuple[str, float, bool]], int]:
    """Return (word, onset_seconds, ends_line) per word, plus the count of one-word lines."""
    words, one_word_lines = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line_words = []
        for mins, secs, text in WORD_TAG.findall(line):
            t = int(mins) * 60 + float(secs.replace(":", "."))
            line_words += [(token, t) for token in text.split()]
        one_word_lines += len(line_words) == 1
        words += [(w, t, i == len(line_words) - 1) for i, (w, t) in enumerate(line_words)]
    return words, one_word_lines


def load_truth(lrc: Path) -> list[tuple[str, float, bool]] | None:
    base = lrc.parent / "groundtruth" / lrc.stem
    txt, csv_path = base.with_name(base.name + ".words.txt"), base.with_name(base.name + ".words.csv")
    if not (txt.is_file() and csv_path.is_file()):
        return None
    tokens = txt.read_text(encoding="utf-8").split()
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(tokens) != len(rows):
        raise ValueError(f"{txt.name}: {len(tokens)} words but {len(rows)} timings")
    return [(w, float(r["word_start"]), r["line_end"] != "nan") for w, r in zip(tokens, rows)]


def score(lrc: Path) -> dict | None:
    truth = load_truth(lrc)
    if truth is None:
        return None
    hyp, one_word_lines = parse_lrc(lrc)

    errors, tp, fp, fn = [], 0, 0, 0
    matcher = SequenceMatcher(a=[norm(w) for w, *_ in truth], b=[norm(w) for w, *_ in hyp], autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            _, ref_t, ref_brk = truth[block.a + k]
            _, hyp_t, hyp_brk = hyp[block.b + k]
            errors.append(abs(hyp_t - ref_t))
            tp += ref_brk and hyp_brk
            fp += hyp_brk and not ref_brk
            fn += ref_brk and not hyp_brk

    n = len(errors)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "song": lrc.stem,
        "recall": n / len(truth) if truth else 0.0,
        "aae": statistics.fmean(errors) if n else float("nan"),
        "median": statistics.median(errors) if n else float("nan"),
        "within_0.3": sum(e <= 0.3 for e in errors) / n if n else 0.0,
        "brk_p": p,
        "brk_r": r,
        "brk_f1": 2 * p * r / (p + r) if p + r else 0.0,
        "one_word": one_word_lines,
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lrc", nargs="+", type=Path)
    args = parser.parse_args()

    rows = [r for r in (score(p) for p in args.lrc) if r]
    if not rows:
        print("no .lrc files with ground truth found", file=sys.stderr)
        return 1

    header = (f"{'song':<30} {'recall':>6} {'AAE':>6} {'median':>6} {'<0.3s':>6}"
              f" {'brk P':>6} {'brk R':>6} {'brk F1':>6} {'1-word':>6}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['song'][:30]:<30} {r['recall']:>6.0%} {r['aae']:>5.2f}s {r['median']:>5.2f}s {r['within_0.3']:>6.0%}"
            f" {r['brk_p']:>6.0%} {r['brk_r']:>6.0%} {r['brk_f1']:>6.0%} {r['one_word']:>6}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
