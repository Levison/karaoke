"""
known_lyrics.py — find a song's lyrics, and work out where each line is sung.

When the lyrics are known, karaoke_lrc.py times those words instead of Whisper's.
Whisper mishears sung words far more often than it misses where the singing is:
on six metal songs, every lyric line it got wrong still had about the right
number of words in about the right place. So Whisper's timed words are used only
to anchor each lyric line to a stretch of the song, and the aligner times the
real words inside it.

Lyrics come from, in order:
  1. <stem>.lyrics.txt next to the audio (or <stem>.txt / <stem>.lyrics.txt in
     --lyrics-dir). Plain text, one lyric line per line.
  2. LRCLIB (lrclib.net, the lyrics database OpenKara uses), searched by the
     audio's artist and title tags and matched on duration. A hit is saved as
     <stem>.lyrics.txt, so a wrong word can be fixed there and the song rerun.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

LRCLIB_SEARCH = "https://lrclib.net/api/search"
USER_AGENT = "karaoke_lrc (https://github.com/Levison/karaoke)"  # LRCLIB asks clients to name themselves
DURATION_SLACK = 3.0  # seconds; a bigger difference is probably another version (live, radio edit)

# Placing lines. Hyp = Whisper's aligned words; a lyric word "anchors" when it
# matches a hyp word.
MIN_ANCHORED = 0.3     # below this share of anchored lyric words, the lyrics are for another song
REPEAT_MIN_WORDS = 4   # unmatched hyp runs at least this long are checked for unwritten repeats
REPEAT_MIN_SHARE = 0.5 # share of a lyric line's words a run must match to count as a repeat of it
PAD = 0.5              # seconds added around a line's anchors
PAD_PER_WORD = 0.6     # plus this per unanchored word at that end of the line


def norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'")).strip("'")


def clean_lyrics(text: str) -> list[str]:
    """Lyric lines as singable text: drops blank lines, [Chorus]-style tags and LRC
    time tags, repeat markers like (x2), and the brackets (not the words) of
    parenthesised backing vocals."""
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", raw)
        line = re.sub(r"\(\s*[x×]\s*\d+\s*\)|(?<!\w)[x×]\d+(?!\w)", " ", line, flags=re.IGNORECASE)
        line = " ".join(line.replace("(", " ").replace(")", " ").split())
        if any(norm(w) for w in line.split()):
            lines.append(line)
    return lines


def fetch_lrclib(artist: str, title: str, duration: float) -> str | None:
    """Plain lyrics of the LRCLIB entry closest in duration, or None. A title
    that finds nothing is retried without suffixes like "(2017 Version)" or
    " - Remastered 2011", which tags carry and LRCLIB titles often don't."""
    if not artist or not title:
        return None
    bare = re.sub(r"\s*[(\[][^)\]]*[)\]]\s*$|\s+-\s+.*$", "", title).strip()
    for name in dict.fromkeys([title, bare]):  # unique, in order
        if name and (text := search_lrclib(artist, name, duration)):
            return text
    return None


def search_lrclib(artist: str, title: str, duration: float) -> str | None:
    query = urllib.parse.urlencode({"artist_name": artist, "track_name": title})
    request = urllib.request.Request(f"{LRCLIB_SEARCH}?{query}", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            results = json.load(response)
    except (OSError, ValueError) as exc:
        print(f"    LRCLIB lookup failed: {exc}", file=sys.stderr)
        return None
    hits = [r for r in results
            if r.get("plainLyrics") and not r.get("instrumental")
            and abs((r.get("duration") or 0) - duration) <= DURATION_SLACK]
    if not hits:
        return None
    return min(hits, key=lambda r: abs(r["duration"] - duration))["plainLyrics"]


def find_lyrics(audio: Path, artist: str, title: str, duration: float,
                lyrics_dir: Path | None, use_lrclib: bool) -> tuple[list[str], str] | None:
    """(lyric lines, where they came from), or None when there are none to use."""
    sidecar = audio.with_name(audio.stem + ".lyrics.txt")
    local = [sidecar]
    if lyrics_dir:
        local = [lyrics_dir / f"{audio.stem}.txt", lyrics_dir / f"{audio.stem}.lyrics.txt"]
    for path in local:
        if path.is_file():
            return clean_lyrics(path.read_text(encoding="utf-8-sig")), path.name

    if use_lrclib and (text := fetch_lrclib(artist, title, duration)):
        sidecar.write_text(text.rstrip() + "\n", encoding="utf-8")
        return clean_lyrics(text), f"LRCLIB, saved as {sidecar.name}"
    return None


@dataclass
class Placement:
    words: list[str]  # the lyric line, split on whitespace
    start: float      # the stretch of song to align it in
    end: float
    anchors: int      # how many of its words matched Whisper's
    first: float = 0.0  # start of its first anchored word
    last: float = 0.0   # end of its last anchored word
    group: int | None = None  # lines sharing a group are aligned together over one stretch


def place_lines(lines: list[str], hyp: list[tuple[str, float, float]]) -> list[Placement] | None:
    """Give each lyric line a stretch of the song, from Whisper's timed words
    (text, start, end). Lines sung more often than the lyrics write them are
    repeated. Returns None if the lyrics barely match what Whisper heard.
    """
    if not hyp:
        return None
    lyric = [(norm(w), li, wi) for li, line in enumerate(lines) for wi, w in enumerate(line.split()) if norm(w)]
    heard = [norm(t) for t, _, _ in hyp]
    matcher = SequenceMatcher(a=[t for t, _, _ in lyric], b=heard, autojunk=False)
    anchors: dict[int, list[tuple[int, int]]] = {}  # line -> [(word index in line, hyp index)]
    used = [False] * len(hyp)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            _, li, wi = lyric[block.a + k]
            anchors.setdefault(li, []).append((wi, block.b + k))
            used[block.b + k] = True
    if sum(map(len, anchors.values())) < MIN_ANCHORED * len(lyric):
        return None

    # (hyp position or None, line index, anchors) in singing order
    order: list[tuple[int | None, int, list[tuple[int, int]]]] = [
        (anchors[li][0][1] if li in anchors else None, li, anchors.get(li, [])) for li in range(len(lines))
    ]
    for pos, li, found in find_repeats(lines, heard, used):
        at = next((k for k, (p, _, _) in enumerate(order) if p is not None and p > pos), len(order))
        order.insert(at, (pos, li, found))

    placements = [line_window(lines[li].split(), found, hyp) for _, li, found in order]
    fill_unanchored(placements, hyp)
    for prev, cur in zip(placements, placements[1:]):
        if prev.anchors and cur.anchors and cur.start < prev.end:
            # Padding overlaps: split the difference, but never cut into anchored words.
            cut = min(max((prev.end + cur.start) / 2, prev.last), max(cur.first, prev.last))
            prev.end, cur.start = max(cut, prev.last), min(cut, cur.first)
    for p in placements:
        p.start = max(p.start, 0.0)
    return placements


def find_repeats(lines: list[str], heard: list[str], used: list[bool]):
    """Yield (hyp position, line index, anchors) for runs of unmatched Whisper
    words that repeat a lyric line: choruses the lyrics write once but the
    singer sings again."""
    candidates = [(li, [norm(w) for w in line.split()]) for li, line in enumerate(lines)]
    candidates = [(li, toks) for li, toks in candidates if len([t for t in toks if t]) >= 2]
    i = 0
    while i < len(heard):
        if used[i]:
            i += 1
            continue
        j = i
        while j < len(heard) and not used[j]:
            j += 1
        pos = i
        while j - i >= REPEAT_MIN_WORDS and pos < j:
            best = None
            for li, toks in candidates:
                window = heard[pos:min(j, pos + len(toks) + 2)]
                m = SequenceMatcher(a=toks, b=window, autojunk=False)
                found = [(b.a + k, pos + b.b + k) for b in m.get_matching_blocks() for k in range(b.size)]
                if len(found) >= REPEAT_MIN_SHARE * len(toks) and (best is None or len(found) > len(best[1])):
                    best = (li, found)
            if best is None:
                pos += 1
                continue
            li, found = best
            yield found[0][1], li, found
            pos = found[-1][1] + 1
        i = j


def line_window(words: list[str], found: list[tuple[int, int]], hyp) -> Placement:
    if not found:
        return Placement(words, 0.0, 0.0, 0)
    (w0, h0), (w1, h1) = found[0], found[-1]
    before, after = w0, len(words) - 1 - w1
    return Placement(words, hyp[h0][1] - PAD - PAD_PER_WORD * before,
                     hyp[h1][2] + PAD + PAD_PER_WORD * after, len(found), hyp[h0][1], hyp[h1][2])


def fill_unanchored(placements: list[Placement], hyp) -> None:
    """Lines Whisper heard nothing of get the whole gap between their anchored
    neighbours, as one group, and the aligner finds where in it they're sung.
    These lines are the weak spot: on JamendoLyrics about half their words land
    over 1 s off. Splitting the gap by word count, matching lines to the vocal
    stem's phrases, and re-transcribing the gap with the lines as Whisper's prompt
    all did no better."""
    song_end = hyp[-1][2] + 5.0
    k = 0
    while k < len(placements):
        if placements[k].anchors:
            k += 1
            continue
        j = k
        while j < len(placements) and not placements[j].anchors:
            j += 1
        gap_start = placements[k - 1].end if k else max(0.0, hyp[0][1] - 5.0)
        gap_end = placements[j].start if j < len(placements) else song_end
        gap_end = max(gap_end, gap_start + 0.5)
        for p in placements[k:j]:
            p.start, p.end, p.group = gap_start, gap_end, k
        k = j
