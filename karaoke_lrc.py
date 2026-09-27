#!/usr/bin/env python3
"""
karaoke_lrc.py — turn an audio file into a word-timed .lrc sidecar for OpenKara.

Pipeline:
    audio -> (optional) Demucs vocal isolation -> WhisperX word timestamps
          -> enhanced LRC with <mm:ss.xx> per-word tags

When the song's lyrics are known (<stem>.lyrics.txt next to the audio, or found
on LRCLIB or Jamendo and saved there), Whisper's words only locate each lyric
line and the real lyrics are timed instead; see known_lyrics.py. Edit
<stem>.lyrics.txt and rerun with --force to fix a wrong word.

The .lrc is written next to the source audio with the same stem. Import both into
OpenKara together: it copies the audio to media/<sha256>.<ext>, so it matches the
.lrc by its [ti:]/[ar:] tags, which are taken from the audio's own tags. An
imported .lrc is cached as manual lyrics, which win over LRCLIB.

Usage:
    uv run karaoke_lrc.py song.mp3
    uv run karaoke_lrc.py *.mp3 --model large-v3
    uv run karaoke_lrc.py song.mp3 --no-separate      # skip Demucs (faster, worse)
    uv run karaoke_lrc.py song.mp3 --plain            # line-level LRC only
    uv run karaoke_lrc.py song.mp3 --keep-stems out/  # save vocals/accompaniment
    uv run karaoke_lrc.py canción.mp3 --language es   # non-English (default: en)
    uv run karaoke_lrc.py song.mp3 --align-model WAV2VEC2_ASR_BASE_960H  # smaller aligner
    uv run karaoke_lrc.py *.mp3 --lyrics-dir lyrics/  # known lyrics as lyrics/<stem>.txt
    uv run karaoke_lrc.py song.mp3 --offline          # only local lyrics files
    uv run karaoke_lrc.py song.mp3 --no-lyrics        # Whisper's words only

Install (on the GPU machine):
    uv sync
    # pyproject.toml pins Python 3.11 and pulls torch/torchaudio/torchvision from
    # the CUDA 12.8 index (PyPI's Windows wheels are CPU-only). ffmpeg must be on PATH.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from known_lyrics import find_lyrics, place_lines

# Line breaking. Every gap between two words gets a break score from the cues
# below, then the whole song is laid out at once to maximise the total score
# while keeping lines near TARGET_LINE_CHARS — the way a word processor wraps a
# paragraph. Measured with eval_lrc.py on JamendoLyrics: 80% line-break F1 vs 45%
# for the old greedy rules, and all 243 nearby settings tried scored 70-81%, so
# these values aren't fragile. Higher LINE_COST = fewer, longer lines.
TARGET_LINE_CHARS = 30
MIN_LINE_CHARS = 10
MAX_LINE_CHARS = 56
LINE_COST = 1.5
SHORT_LINE_PENALTY = 3.0
BREAK_CAPITAL = 2.0       # next word capitalised: Whisper capitalises lyric-line starts
BREAK_SEGMENT = 2.0       # WhisperX segment boundary
BREAK_SENTENCE = 1.5      # . ? !
BREAK_CLAUSE = 0.5        # , ; :
BREAK_GAP_PER_SEC = 1.0   # per second of silence, capped at 2 s. Held notes make pauses a weak cue
NO_BREAK_PENALTY = 3.0    # breaking right after a NO_BREAK_AFTER word
NO_BREAK_AFTER = {"a", "an", "the", "my", "your", "his", "her", "our", "their", "its", "of", "to",
                  "mr.", "mrs.", "ms.", "dr.", "st."}
ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.", "vs."}  # "." that doesn't end a sentence
FIRST_PERSON = {"i", "i'm", "i'll", "i've", "i'd"}  # capitalised anywhere, so not a line-start cue

# Forced-alignment model per language; languages not listed use WhisperX's default.
# WhisperX's English default (WAV2VEC2_ASR_BASE_960H) crams repeated lines ("I want
# you to feel" x4) into the first repeat. The large model keeps them apart: on that
# JamendoLyrics song, onsets >1 s off drop 7% -> 1% and mean error 0.39 -> 0.11 s;
# the other songs are unchanged. Costs a one-time 1.26 GB download.
ALIGN_MODELS = {"en": "WAV2VEC2_ASR_LARGE_LV60K_960H"}

# Line starts pulled into the previous note (see retime_line_starts). A pause is the
# vocal stem PAUSE_DB below the song's loud level for MIN_PAUSE; Demucs bleed means
# breaths are 12-30 dB dips, not silence. On 20 JamendoLyrics songs this moved 16
# words: 15 closer to the true onset, 1 further. Neighbouring settings (10-15 dB,
# 0.2-0.4 s) behaved the same.
PAUSE_DB = 12.0
MIN_PAUSE = 0.25
MAX_VOICE_BEFORE = 0.3   # voice the word has of its own before the pause
MIN_VOICE_AFTER = 0.1    # unclaimed voice after the pause, before the next word

# Dropped repeats (see find_unheard). Whisper sometimes writes a repeated line fewer
# times than it's sung ("I want you to feel" x2 for x4 sung). Voice that comes back
# after a pause and runs MIN_UNHEARD seconds with no word in the transcript is
# transcribed again on its own. The new words are kept only if they repeat a
# 4-word phrase from elsewhere in the song and aren't mostly VOCABLES. On 20
# JamendoLyrics songs that added 24 words, all real lyrics, and rejected 56
# (invented lines like "I'm a falcon!", and "do do do" / "ah ah" fills).
MIN_UNHEARD = 1.5
VOCABLES = {"oh", "ah", "ooh", "oo", "do", "doo", "da", "la", "na", "whoa", "woah", "uh", "mm", "hmm", "hey", "ha", "eh"}


@dataclass
class Word:
    text: str
    start: float
    end: float
    segment: int = 0  # index of the WhisperX segment the word came from


def fmt_time(seconds: float) -> str:
    """Format seconds as mm:ss.xx (centiseconds), as LRC expects."""
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100))
    minutes, rem_cs = divmod(total_cs, 6000)
    secs, cs = divmod(rem_cs, 100)
    return f"{minutes:02d}:{secs:02d}.{cs:02d}"


def break_score(word: Word, nxt: Word) -> float:
    """How strongly the transcript and timing suggest a line break between two words."""
    stripped = word.text.rstrip('"\')]}').lower()
    if stripped.rstrip(",") in NO_BREAK_AFTER:
        return -NO_BREAK_PENALTY

    score = BREAK_GAP_PER_SEC * min(max(nxt.start - word.end, 0.0), 2.0)
    if nxt.text[:1].isupper() and nxt.text.lower().strip(",.?!") not in FIRST_PERSON:
        score += BREAK_CAPITAL
    if nxt.segment != word.segment:
        score += BREAK_SEGMENT
    if stripped.endswith((".", "?", "!")) and stripped not in ABBREVIATIONS:
        score += BREAK_SENTENCE
    elif stripped.endswith((",", ";", ":")):
        score += BREAK_CLAUSE
    return score


def group_into_lines(words: list[Word]) -> list[list[Word]]:
    """Group a flat word stream into singable lines.

    Picks the set of breaks that minimises, summed over lines: LINE_COST, the
    squared distance from TARGET_LINE_CHARS, and minus the break score where the
    line ends. Dynamic programming over "best layout of the first k words".
    """
    n = len(words)
    if n == 0:
        return []
    scores = [break_score(words[i], words[i + 1]) for i in range(n - 1)] + [0.0]

    best = [0.0] + [math.inf] * n  # best[k]: lowest cost for laying out words[:k]
    line_start = [0] * (n + 1)     # line_start[k]: where the last line of that layout begins
    for k in range(1, n + 1):
        chars = -1
        for j in range(k - 1, -1, -1):  # candidate last line: words[j:k]
            chars += len(words[j].text) + 1
            if chars > MAX_LINE_CHARS and j < k - 1:
                break
            cost = LINE_COST + ((chars - TARGET_LINE_CHARS) / TARGET_LINE_CHARS) ** 2 - scores[k - 1]
            if chars < MIN_LINE_CHARS and k < n:
                cost += SHORT_LINE_PENALTY
            if best[j] + cost < best[k]:
                best[k], line_start[k] = best[j] + cost, j

    lines: list[list[Word]] = []
    k = n
    while k > 0:
        lines.append(words[line_start[k]:k])
        k = line_start[k]
    return lines[::-1]


def build_lrc(
    lines: list[list[Word]],
    *,
    title: str | None = None,
    artist: str | None = None,
    plain: bool = False,
) -> str:
    """Render grouped words as an enhanced (word-timed) LRC document."""
    out: list[str] = []
    if title:
        out.append(f"[ti:{title}]")
    if artist is not None:  # an empty [ar:] still matches an untagged song in OpenKara
        out.append(f"[ar:{artist}]")
    out.append("[by:karaoke_lrc.py]")
    out.append("")

    for line in lines:
        if not line:
            continue
        stamp = f"[{fmt_time(line[0].start)}]"
        if plain:
            out.append(stamp + " ".join(w.text for w in line))
        else:
            parts = []
            for i, w in enumerate(line):
                trailing = "" if i == len(line) - 1 else " "
                parts.append(f"<{fmt_time(w.start)}>{w.text}{trailing}")
            out.append(stamp + "".join(parts))

    return "\n".join(out) + "\n"


def separate_vocals(audio: Path, workdir: Path, model: str = "htdemucs") -> Path:
    """Run Demucs and return the isolated vocal stem."""
    if importlib.util.find_spec("demucs") is None:
        raise RuntimeError("demucs not installed — run `uv sync`, or pass --no-separate")

    # Run via this interpreter rather than a `demucs` on PATH, which only exists
    # when the venv is activated (and activation is often blocked on Windows).
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", model,
        # The default (1 shift) offsets the input by a random 0-0.5 s and doesn't
        # average anything, so every run gives a slightly different stem — and
        # occasionally a different alignment. 0 makes output reproducible.
        "--shifts", "0",
        "-o", str(workdir),
        str(audio),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-2000:] + "\n")
        raise RuntimeError(f"demucs failed with exit code {result.returncode}")

    matches = list(workdir.glob(f"{model}/**/vocals.*"))
    if not matches:
        raise RuntimeError(f"demucs produced no vocals stem under {workdir}")
    return matches[0]


def import_whisperx():
    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError("whisperx not installed — run `uv sync`") from exc
    return whisperx


class Transcriber:
    """WhisperX transcription + forced alignment. Models load on first use.

    Keep only one of Whisper and the aligner loaded: transcribe every file, then
    unload_whisper(), then align (and unload_aligners() before transcribing again).
    Whisper large-v2 holds ~4.5 GB of VRAM and the large English aligner ~2 GB at
    peak; together they overflow an 8 GB card into shared memory and alignment
    runs ~20x slower.
    """

    def __init__(self, model_name: str, device: str, language: str | None, align_model: str | None = None):
        self.whisperx = import_whisperx()
        self.device = device
        self.language = language
        self.model_name = model_name
        self.align_model = align_model
        self.model = None
        self.align_models: dict[str, tuple] = {}
        self.load_whisper()  # fail early if the model can't load

    def load_whisper(self) -> None:
        if self.model is None:
            compute_type = "float16" if self.device == "cuda" else "int8"
            self.model = self.whisperx.load_model(
                self.model_name, self.device, compute_type=compute_type, language=self.language,
            )

    def transcribe(self, audio_data) -> dict:
        """Transcribe 16 kHz mono samples into WhisperX segments."""
        self.load_whisper()
        return self.model.transcribe(audio_data, batch_size=16, language=self.language)

    def transcribe_window(self, audio_data, start: float, end: float) -> dict:
        """Transcribe audio_data[start:end] (seconds) on its own; segment times are song times."""
        result = self.transcribe(audio_data[int(start * 16000):int(end * 16000)])
        for segment in result["segments"]:
            segment["start"] += start
            segment["end"] += start
        return result

    def unload_whisper(self) -> None:
        self.model = None
        self._free_gpu()

    def unload_aligners(self) -> None:
        self.align_models.clear()
        self._free_gpu()

    @staticmethod
    def _free_gpu() -> None:
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()

    def aligner(self, lang: str) -> tuple:
        if lang not in self.align_models:
            self.align_models[lang] = self.whisperx.load_align_model(
                language_code=lang, device=self.device, model_name=self.align_model or ALIGN_MODELS.get(lang),
            )
        return self.align_models[lang]

    def align(self, result: dict, audio_data) -> list[Word]:
        """Force-align a transcript against its samples, returning word-level timings."""
        align_model, metadata = self.aligner(result.get("language", self.language or "en"))
        aligned = self.whisperx.align(
            result["segments"], align_model, metadata, audio_data, self.device,
            return_char_alignments=False,
        )

        words: list[Word] = []
        for seg_index, segment in enumerate(aligned.get("segments", [])):
            for w in segment.get("words", []):
                text = (w.get("word") or "").strip()
                start, end = w.get("start"), w.get("end")
                if not text or start is None or end is None:
                    # Alignment drops timings for some tokens (numerals, odd glyphs).
                    continue
                words.append(Word(text=text, start=float(start), end=float(end), segment=seg_index))
        return words

    def align_line(self, text: str, start: float, end: float, audio_data) -> list[tuple[float, float] | None]:
        """Time each whitespace-separated word of text within [start, end] seconds;
        None for words the aligner couldn't place."""
        align_model, metadata = self.aligner(self.language or "en")
        segment = {"text": text, "start": start, "end": min(end, len(audio_data) / 16000)}
        aligned = self.whisperx.align([segment], align_model, metadata, audio_data, self.device,
                                      return_char_alignments=False)
        # WhisperX may split a line at sentence ends, and returns no words for a line
        # it fails to align; words keep their order either way.
        found = [w for seg in aligned.get("segments", []) for w in seg.get("words", [])]
        if len(found) != len(text.split()):
            return [None] * len(text.split())
        return [(float(w["start"]), float(w["end"])) if w.get("start") is not None and w.get("end") is not None
                else None for w in found]


def vocal_loudness(samples) -> "np.ndarray":
    """Level per 10 ms frame (20 ms window) of 16 kHz samples, in dB relative to the
    song's loud level (its 95th percentile)."""
    import numpy as np

    frames = samples[: len(samples) // 160 * 160].reshape(-1, 160).astype(np.float64)
    power = (frames ** 2).mean(axis=1)
    db = 10 * np.log10((power[:-1] + power[1:]) / 2 + 1e-18)
    return db - np.percentile(db, 95)


def retime_line_starts(words: list[Word], vocals, line_starts: set[int] | None = None) -> int:
    """Move line-starting words the aligner dragged into the previous held note.

    The aligner sometimes starts a line's first word inside the previous word's
    held vowel ("...feel / I want you to feel": the "I" lands 1-2 s early). In the
    vocal stem, such a word has almost no voice of its own before a pause, and after
    the pause there's voice that no word claims. Only capitalised words (Whisper's
    line starts) are moved: a line's last word is often short and followed by a
    breath too, and moving those was wrong more often than right. With known lyrics,
    line_starts gives the indices of each lyric line's first word instead.
    Returns the number of words moved.
    """
    loud = vocal_loudness(vocals) > -PAUSE_DB
    moved = 0
    for i, (word, nxt) in enumerate(zip(words, words[1:])):
        a, b = int(word.start * 100), int(nxt.start * 100)
        starts_line = i in line_starts if line_starts is not None else word.text[:1].isupper()
        if not starts_line or b - a < 3 or b > len(loud):
            continue
        span = loud[a:b]
        runs = pauses(span)
        if not runs:
            continue
        p0, p1 = runs[-1]
        if span[:p0].sum() / 100 <= MAX_VOICE_BEFORE and span[p1:].sum() / 100 >= MIN_VOICE_AFTER:
            duration = word.end - word.start
            word.start = (a + p1) / 100
            word.end = min(word.start + duration, nxt.start)
            moved += 1
    return moved


def pauses(loud) -> list[tuple[int, int]]:
    """(start, end) frame ranges of quiet runs >= MIN_PAUSE that end before the span does."""
    runs, j = [], 0
    while j < len(loud):
        r = j
        while r < len(loud) and not loud[r]:
            r += 1
        if r > j and (r - j) / 100 >= MIN_PAUSE and r < len(loud):
            runs.append((j, r))
        j = max(r, j + 1)
    return runs


def find_unheard(words: list[Word], vocals) -> list[tuple[float, float, int]]:
    """Stretches of singing the transcript has no words for, as (start, end, i): the
    stretch sits between words[i] and words[i + 1].

    A held note ends at the singer's next pause, so voice that comes back after a
    pause and runs MIN_UNHEARD seconds before the next word starts is unaccounted for.
    """
    loud = vocal_loudness(vocals) > -PAUSE_DB
    found = []
    for i, (word, nxt) in enumerate(zip(words, words[1:])):
        a, b = int(word.start * 100), min(int(nxt.start * 100), len(loud))
        span = loud[a:b]
        runs = pauses(span)
        if not runs or span[runs[0][1]:].sum() / 100 < MIN_UNHEARD:
            continue
        start = max((a + runs[0][1]) / 100 - 0.3, word.end)  # a little lead-in before the voice returns
        end = nxt.start - 0.05
        if end - start >= 1.0:
            found.append((start, end, i))
    return found


def norm_word(text: str) -> str:
    return re.sub(r"[^a-z0-9']", "", text.lower()).strip("'")


def repeats_song(new: list[Word], words: list[Word], n: int = 4) -> bool:
    """Whether re-transcribed words look like a dropped repeat rather than an invention:
    at least half of them sit in an n-word phrase found elsewhere in the song, and
    they aren't mostly vocables."""
    song = [norm_word(w.text) for w in words]
    phrases = {tuple(song[i:i + n]) for i in range(len(song) - n + 1)}
    toks = [norm_word(w.text) for w in new]
    covered = [False] * len(toks)
    for i in range(len(toks) - n + 1):
        if tuple(toks[i:i + n]) in phrases:
            covered[i:i + n] = [True] * n
    vocables = sum(t in VOCABLES for t in toks)
    return bool(toks) and sum(covered) >= len(toks) / 2 and vocables < 0.8 * len(toks)


def prepare(audio: Path, args: argparse.Namespace, workdir: Path):
    """Separate vocals (unless disabled) and decode to the 16 kHz mono WhisperX expects.

    Returns (samples, whether they're an isolated vocal stem)."""
    target = audio
    if not args.no_separate:
        try:
            target = separate_vocals(audio, workdir, args.demucs_model)
            if args.keep_stems:
                dest = Path(args.keep_stems) / audio.stem
                dest.mkdir(parents=True, exist_ok=True)
                for stem in target.parent.glob("*.*"):
                    shutil.copy2(stem, dest / stem.name)
                print(f"    stems saved to {dest}")
        except RuntimeError as exc:
            print(f"    separation failed ({exc}) — falling back to original mix")
            target = audio

    samples = import_whisperx().load_audio(str(target))
    shutil.rmtree(workdir, ignore_errors=True)  # stems are ~40 MB each; don't pile them up
    return samples, target != audio


def fill_unheard(aligned: list[tuple], transcriber: Transcriber) -> None:
    """Re-transcribe singing the transcript missed (see find_unheard) and splice in
    the words that pass repeats_song. aligned holds (audio, samples, is_vocals, words)
    per file; words are updated in place.

    Runs once over every file, since it needs Whisper back after alignment.
    """
    gaps = [(n, *gap) for n, (_, samples, is_vocals, words) in enumerate(aligned) if is_vocals
            for gap in find_unheard(words, samples)]
    if not gaps:
        return
    print(f"\nRe-transcribing {len(gaps)} stretch(es) of singing with no words")
    transcriber.unload_aligners()
    heard = []
    for n, start, end, i in gaps:
        try:
            heard.append(transcriber.transcribe_window(aligned[n][1], start, end))
        except Exception as exc:  # noqa: BLE001
            print(f"    {aligned[n][0].name}: failed: {exc}", file=sys.stderr)
            heard.append({"segments": []})
    transcriber.unload_whisper()

    inserts: dict[int, list[tuple[int, list[Word]]]] = {}
    for g, ((n, start, end, i), result) in enumerate(zip(gaps, heard)):
        audio, samples, _, words = aligned[n]
        if not result["segments"]:
            continue
        try:
            new = [w for w in transcriber.align(result, samples) if start - 0.2 <= w.start < end]
        except Exception as exc:  # noqa: BLE001
            print(f"    {audio.name}: failed: {exc}", file=sys.stderr)
            continue
        if new and repeats_song(new, words):
            for w in new:
                w.segment = -1 - g  # a segment of its own, for line breaking
            inserts.setdefault(n, []).append((i, new))
            print(f"    {audio.name} {fmt_time(start)}: +{len(new)} words: {' '.join(w.text for w in new)}")
    for n, items in inserts.items():
        _, samples, _, words = aligned[n]
        for i, new in sorted(items, key=lambda item: -item[0]):  # back to front keeps indices valid
            words[i + 1:i + 1] = new
        retime_line_starts(words, samples)  # new words can land in the previous note too


def spread(times: list[tuple[float, float] | None], lo: float, hi: float) -> list[tuple[float, float]]:
    """Fill in untimed words: each run shares the time between its timed
    neighbours (or lo/hi at the ends) evenly."""
    out = list(times)
    k = 0
    while k < len(out):
        if out[k] is not None:
            k += 1
            continue
        j = k
        while j < len(out) and out[j] is None:
            j += 1
        a = out[k - 1][1] if k else lo
        b = out[j][0] if j < len(out) else max(hi, a)
        step = max(b - a, 0.0) / (j - k)
        for n in range(k, j):
            out[n] = (a + step * (n - k), a + step * (n - k + 1))
        k = j
    return out


def time_lyrics(placements, transcriber: Transcriber, samples) -> list[list[Word]]:
    """Align each placed lyric line (see known_lyrics.place_lines) in its stretch of
    the song. Returns the lines as timed words."""
    # Consecutive placements in the same group are aligned as one text.
    batches: list[list[int]] = []
    for li, p in enumerate(placements):
        if batches and p.group is not None and placements[batches[-1][0]].group == p.group:
            batches[-1].append(li)
        else:
            batches.append([li])

    lines, last_start = [], 0.0
    for batch in batches:
        p = placements[batch[0]]
        words = [w for li in batch for w in placements[li].words]
        text = " ".join(words)
        try:
            times = transcriber.align_line(text, p.start, p.end, samples)
        except Exception as exc:  # noqa: BLE001 — one line shouldn't sink the song
            print(f"    couldn't align \"{text}\": {exc}", file=sys.stderr)
            times = [None] * len(words)
        lo, hi = (p.first, p.last) if p.anchors and all(t is None for t in times) else (p.start, p.end)
        timed = iter(spread(times, lo, hi))
        for li in batch:
            line = []
            for w in placements[li].words:
                start, end = next(timed)
                start = max(start, last_start)  # keep the song's words in order
                line.append(Word(text=w, start=start, end=max(end, start), segment=li))
                last_start = start
            lines.append(line)
    return lines


def write_lrc(audio: Path, words: list[Word], args: argparse.Namespace,
              lines: list[list[Word]] | None = None) -> bool:
    """Write the .lrc; lines are the lyric lines when the lyrics are known,
    otherwise words are laid out by group_into_lines."""
    if not words:
        print("    no words transcribed — is this an instrumental?")
        return False

    from fix_lrc_tags import read_tags

    lines = lines or group_into_lines(words)
    title, artist = read_tags(audio)
    out_path = audio.with_suffix(".lrc")
    out_path.write_text(build_lrc(lines, title=title, artist=artist, plain=args.plain), encoding="utf-8")
    print(f"    wrote {out_path.name} — {len(lines)} lines, {len(words)} words")
    if "]" in title or "]" in artist:
        print("    ']' in the title/artist: OpenKara can't auto-match this one, use Edit lyrics there")
    return True


def lookup_lyrics(audio: Path, duration: float, args: argparse.Namespace) -> list[str] | None:
    """Known lyric lines for audio (see known_lyrics.find_lyrics), or None."""
    if args.no_lyrics:
        return None
    from fix_lrc_tags import read_tags

    title, artist = read_tags(audio)
    found = find_lyrics(audio, artist, title, duration, args.lyrics_dir, not args.offline)
    if not found or not found[0]:
        print("    no known lyrics, using Whisper's words")
        return None
    lines, source = found
    print(f"    lyrics: {len(lines)} lines from {source}")
    return lines


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> int:
    # Redirected output on Windows defaults to the ANSI codepage, which can't
    # encode non-Latin song titles. Force UTF-8 so a print never kills a run.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="+", type=Path, help="audio file(s) to process")
    parser.add_argument("--model", default="large-v2", help="Whisper model (default: large-v2)")
    parser.add_argument("--demucs-model", default="htdemucs", help="Demucs model (default: htdemucs)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--language", default="en",
        help="language code (default: en), or 'auto' to detect. Detection only hears the "
             "first 30 s, so an instrumental intro can make it guess wrong",
    )
    parser.add_argument(
        "--align-model", metavar="NAME",
        help="forced-alignment model: a torchaudio bundle or Hugging Face wav2vec2 id "
             "(default: WAV2VEC2_ASR_LARGE_LV60K_960H for English, WhisperX's default otherwise)",
    )
    parser.add_argument("--no-separate", action="store_true", help="skip Demucs vocal isolation")
    parser.add_argument("--plain", action="store_true", help="line-level LRC instead of word-timed")
    parser.add_argument("--keep-stems", metavar="DIR", help="save separated stems to DIR")
    parser.add_argument("--force", action="store_true", help="overwrite existing .lrc")
    parser.add_argument(
        "--lyrics-dir", metavar="DIR", type=Path,
        help="read known lyrics from DIR/<stem>.txt or DIR/<stem>.lyrics.txt "
             "(default: <stem>.lyrics.txt next to the audio)",
    )
    parser.add_argument("--offline", action="store_true",
                        help="don't look lyrics up online (LRCLIB, and Jamendo if JAMENDO_CLIENT_ID is set)")
    parser.add_argument("--no-lyrics", action="store_true",
                        help="ignore known lyrics and use Whisper's words (the old behaviour)")
    args = parser.parse_args()

    args.device = resolve_device(args.device)
    if args.language == "auto":
        args.language = None

    files = [p for p in args.audio if p.is_file()]
    missing = [p for p in args.audio if not p.is_file()]
    for p in missing:
        print(f"not a file, skipping: {p}", file=sys.stderr)
    if not files:
        print("nothing to do", file=sys.stderr)
        return 1

    todo = []
    for audio in files:
        if audio.with_suffix(".lrc").exists() and not args.force:
            print(f"skipping {audio.name} — .lrc exists (use --force to overwrite)")
        else:
            todo.append(audio)

    # Phases rather than file-by-file: Demucs (a child process), Whisper and the
    # aligner each hold GBs of VRAM. Interleaving Demucs with a loaded Whisper
    # overflowed an 8 GB card into shared system memory and made Demucs ~4x
    # slower. Running each phase over every file also loads each model only once.
    jobs = []
    stage = "Decoding audio" if args.no_separate else f"Separating vocals with Demucs ({args.demucs_model})"
    print(f"\n{stage}")
    with tempfile.TemporaryDirectory() as tmp:
        for i, audio in enumerate(todo, 1):
            print(f"  [{i}/{len(todo)}] {audio.name}")
            try:
                samples, is_vocals = prepare(audio, args, Path(tmp) / str(i))
            except Exception as exc:  # noqa: BLE001 — one bad file shouldn't kill the batch
                print(f"    failed: {exc}", file=sys.stderr)
                continue
            jobs.append((audio, samples, is_vocals, lookup_lyrics(audio, len(samples) / 16000, args)))

    written = 0
    if jobs:
        print(f"\nTranscribing with WhisperX ({args.model}, {args.device}, language: {args.language or 'auto'})")
        try:
            transcriber = Transcriber(args.model, args.device, args.language, args.align_model)
        except Exception as exc:  # noqa: BLE001
            print(f"  failed to load WhisperX: {exc}", file=sys.stderr)
            return 1
        transcripts = []
        for i, (audio, samples, is_vocals, lyrics) in enumerate(jobs, 1):
            print(f"  [{i}/{len(jobs)}] {audio.name}")
            try:
                transcripts.append((audio, samples, is_vocals, lyrics, transcriber.transcribe(samples)))
            except Exception as exc:  # noqa: BLE001
                print(f"    failed: {exc}", file=sys.stderr)

        transcriber.unload_whisper()
        aligner = args.align_model or (
            ALIGN_MODELS.get(args.language, "WhisperX default") if args.language else "per detected language"
        )
        print(f"\nAligning words ({aligner})")
        aligned = []
        for i, (audio, samples, is_vocals, lyrics, result) in enumerate(transcripts, 1):
            print(f"  [{i}/{len(transcripts)}] {audio.name}")
            try:
                words = transcriber.align(result, samples)
                lines = None
                if lyrics:
                    placements = place_lines(lyrics, [(w.text, w.start, w.end) for w in words])
                    if placements is None:
                        print("    the lyrics barely match what's sung (another song?), using Whisper's words")
                    else:
                        lines = time_lyrics(placements, transcriber, samples)
                        words = [w for line in lines for w in line]
                        extra = len(placements) - len(lyrics)
                        print(f"    timed the lyrics: {len(lines)} lines"
                              + (f", {extra} sung more often than written" if extra else ""))
                if is_vocals:  # in a full mix the band plays through the singer's pauses
                    starts = None
                    if lines:
                        starts, k = set(), 0
                        for line in lines:
                            starts.add(k)
                            k += len(line)
                    if moved := retime_line_starts(words, samples, starts):
                        print(f"    moved {moved} line start(s) out of the previous note")
                aligned.append((audio, samples, is_vocals, words, lines))
            except Exception as exc:  # noqa: BLE001
                print(f"    failed: {exc}", file=sys.stderr)

        # Known lyrics already hold every repeat; re-transcribing only helps Whisper's words.
        fill_unheard([job[:4] for job in aligned if job[4] is None], transcriber)
        for audio, _, _, words, lines in aligned:
            if write_lrc(audio, words, args, lines):
                written += 1

    print(f"\ndone — {written}/{len(files)} file(s) written")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
