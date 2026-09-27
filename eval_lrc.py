#!/usr/bin/env python3
"""
eval_lrc.py — score a generated enhanced .lrc against JamendoLyrics ground truth.

    uv run eval_lrc.py test_songs/*.lrc          # skips files with no ground truth
    uv run eval_lrc.py --classic test_songs/*.lrc

Ground truth is looked up as groundtruth/<stem>.words.txt + .words.csv next to
the .lrc (the layout JamendoLyrics uses: one word per whitespace token, one CSV
row per word with word_start/word_end in seconds, and line_end set on the last
word of each lyric line).

The default report scores what a singer sees. It never pairs .lrc words with
reference words one to one: it asks, for each word when it's sung, whether the
screen highlights that word then. Pairing is what the classic metrics do, and
it misleads here: when the .lrc has one chorus more or less than the song, the
pairing matches whole choruses to the wrong copy and reports seconds of error
for lines that are on time (Songwriterz: 6.9 s mean error, 98% of lines on time).

    on-time    sung words the .lrc highlights (same word) from 0.3 s early to
               0.2 s late. Users notice late lyrics sooner than early ones:
               Lizé Masclef, Vaglio & Moussallam, "User-centered evaluation of
               lyrics-to-audio alignment", ISMIR 2021, put the points where half
               of listeners notice at about -0.3 s and +0.2 s.
    <=1s       sung words highlighted within 1 s either way: readable, if off
    bad lines  sung lines with under half their words within 1 s: visibly
               broken lines, the thing to minimise
    stray      .lrc words no one sings (same word) within 1 s of that time:
               invented words, mishearings, lines placed in the wrong spot
    starts P/R line starts: .lrc lines starting within 0.5 s of a sung line's
               start, and sung lines with an .lrc line starting that close

--classic prints the older, pairing-based metrics:
    recall     matched reference words / all reference words
    AAE        mean absolute onset error in seconds, over matched words
               (the standard lyrics-alignment metric)
    median     median onset error
    <0.3s      share of matched words whose onset is within 0.3 s
    brk P/R/F1 line breaks: precision (LRC breaks that are real lyric-line
               ends), recall (lyric-line ends the LRC also breaks at), and F1;
               "-" when the ground truth has no line ends
    1-word     LRC lines holding a single word (usually a bad break)
"""

from __future__ import annotations

import argparse
import bisect
import csv
import re
import statistics
import sys
from difflib import SequenceMatcher
from pathlib import Path

WORD_TAG = re.compile(r"<(\d+):(\d+(?:[.:]\d+)?)>([^<]*)")

EARLY, LATE = 0.3, 0.2  # on-time window: highlighted up to EARLY s before the word is sung, LATE s after
NEAR = 1.0              # "readable" window, either way
LINE_START = 0.5        # line starts this close count as the same


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
    has_lines = any(brk for *_, brk in truth)  # False when the ground truth has no line times
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "song": lrc.stem,
        "recall": n / len(truth) if truth else 0.0,
        "aae": statistics.fmean(errors) if n else float("nan"),
        "median": statistics.median(errors) if n else float("nan"),
        "within_0.3": sum(e <= 0.3 for e in errors) / n if n else 0.0,
        "brk_p": p if has_lines else None,
        "brk_r": r if has_lines else None,
        "brk_f1": (2 * p * r / (p + r) if p + r else 0.0) if has_lines else None,
        "one_word": one_word_lines,
    }


def ux_score(lrc: Path) -> dict | None:
    """The default, pairing-free report (see the module docstring)."""
    truth = load_truth(lrc)
    if truth is None:
        return None
    hyp, _ = parse_lrc(lrc)

    def onsets(words) -> dict[str, list[float]]:
        table: dict[str, list[float]] = {}
        for w, t, _ in words:
            table.setdefault(norm(w), []).append(t)
        for times in table.values():
            times.sort()
        return table

    shown, sung = onsets(hyp), onsets(truth)

    def offsets(word: str, t: float, table) -> list[float]:
        """(onset - t) for each onset of word in table within NEAR of t."""
        times = table.get(norm(word), [])
        i, j = bisect.bisect_left(times, t - NEAR), bisect.bisect_right(times, t + NEAR)
        return [x - t for x in times[i:j]]

    on_time = near = 0
    lines, line = [], []  # per sung line: whether each word is shown within NEAR
    for w, t, ends in truth:
        found = offsets(w, t, shown)
        on_time += any(-EARLY <= d <= LATE for d in found)
        near += bool(found)
        line.append(bool(found))
        if ends:
            lines.append(line)
            line = []
    if line:
        lines.append(line)
    has_lines = any(ends for *_, ends in truth)
    stray = sum(not offsets(w, t, sung) for w, t, _ in hyp)

    def starts(words) -> list[float]:
        out, new = [], True
        for _, t, ends in words:
            if new:
                out.append(t)
            new = ends
        return out

    def hits(a: list[float], b: list[float]) -> float:
        return sum(any(abs(x - y) <= LINE_START for y in b) for x in a) / len(a) if a else 0.0

    hyp_starts, true_starts = starts(hyp), starts(truth)
    return {
        "song": lrc.stem,
        "on_time": on_time / len(truth),
        "near": near / len(truth),
        "bad_lines": sum(sum(l) < len(l) / 2 for l in lines) if has_lines else None,
        "lines": len(lines) if has_lines else None,
        "stray": stray / len(hyp) if hyp else 0.0,
        "start_p": hits(hyp_starts, true_starts) if has_lines else None,
        "start_r": hits(true_starts, hyp_starts) if has_lines else None,
    }


def print_ux(rows: list[dict]) -> int:
    if not rows:
        print("no .lrc files with ground truth found", file=sys.stderr)
        return 1

    def pct(v) -> str:
        return f"{'-':>8}" if v is None else f"{v:>8.0%}"

    def mean(key: str):
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.fmean(vals) if vals else None

    header = f"{'song':30} {'on-time':>8} {'<=1s':>8} {'bad lines':>10} {'stray':>8} {'starts P':>8} {'starts R':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        bad = "-" if r["bad_lines"] is None else f"{r['bad_lines']}/{r['lines']}"
        print(f"{r['song'][:30]:30} {pct(r['on_time'])} {pct(r['near'])} {bad:>10} {pct(r['stray'])}"
              f" {pct(r['start_p'])} {pct(r['start_r'])}")
    print("-" * len(header))
    bad_total = sum(r["bad_lines"] or 0 for r in rows)
    line_total = sum(r["lines"] or 0 for r in rows)
    print(f"{'mean over songs':30} {pct(mean('on_time'))} {pct(mean('near'))} {f'{bad_total}/{line_total}':>10}"
          f" {pct(mean('stray'))} {pct(mean('start_p'))} {pct(mean('start_r'))}")
    worst = min(rows, key=lambda r: r["on_time"])
    with_lines = [r for r in rows if r["bad_lines"] is not None]
    clean = sum(r["bad_lines"] == 0 for r in with_lines)
    print(f"worst song: {worst['song']} ({worst['on_time']:.0%} on time); "
          f"songs with no bad lines: {clean}/{len(with_lines)}")
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lrc", nargs="+", type=Path)
    parser.add_argument("--classic", action="store_true", help="the older, pairing-based metrics")
    args = parser.parse_args()
    if not args.classic:
        return print_ux([r for r in (ux_score(p) for p in args.lrc) if r])

    rows = [r for r in (score(p) for p in args.lrc) if r]
    if not rows:
        print("no .lrc files with ground truth found", file=sys.stderr)
        return 1

    header = (f"{'song':<30} {'recall':>6} {'AAE':>6} {'median':>6} {'<0.3s':>6}"
              f" {'brk P':>6} {'brk R':>6} {'brk F1':>6} {'1-word':>6}")
    print(header)
    print("-" * len(header))
    pct = lambda v: f"{'-':>6}" if v is None else f"{v:>6.0%}"
    for r in rows:
        print(
            f"{r['song'][:30]:<30} {r['recall']:>6.0%} {r['aae']:>5.2f}s {r['median']:>5.2f}s {r['within_0.3']:>6.0%}"
            f" {pct(r['brk_p'])} {pct(r['brk_r'])} {pct(r['brk_f1'])} {r['one_word']:>6}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
