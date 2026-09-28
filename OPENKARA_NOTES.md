# OpenKara scoping notes — karaoke lyrics pipeline

From reading `thedavidweng/OpenKara` @ main (Apache-2.0, Tauri 2 + Rust + React),
`src-tauri/src/lyrics/` (`fetch.rs`, `parser.rs`, `acquisition.rs`).

## Getting our lyrics into OpenKara (no fork needed)

Acquisition order: cache → embedded tags → TTML/LYS/LRC sidecar → AMLL → LRCLIB
→ LrcApi. An imported `.lrc` is cached as `Manual` lyrics, so it beats LRCLIB.

Import the `.lrc` **in the same import as the audio** (checked against build
74240930, 2026-09-19). The sidecar step never sees our files: import copies audio
to `<library>/media/<sha256>.<ext>` and looks for `<sha256>.lrc`. Instead the
frontend passes the `.lrc` to `import_lyrics_files`, which matches it to a song by
`[ar:]` + `[ti:]`, both equal to the audio's artist and title (case-insensitive;
title falls back to the file name; an empty `[ar:]` matches an untagged song, a
missing one matches nothing). `karaoke_lrc.py` writes both from the audio's tags;
`uv run fix_lrc_tags.py` fixes older files. An unmatched `.lrc` shows an error.

- Brackets in title/artist (*Vision [Radio Edit]*) truncate at the first `]`
  upstream, so those match nothing. Fixed on our branch
  `fix/lyrics-bracketed-metadata-tags` (upstream issue #465, PR #466); otherwise
  use Edit lyrics in OpenKara.
- Keep the library folder outside the song folders: folder import scans 3 levels
  deep and would pick up `media/` copies.

Enhanced LRC gives word-by-word highlighting (`parser.rs::parse_word_tokens`):
`[00:12.00]<00:12.00>I <00:12.30>see <00:12.60>trees`. A line tag is still
required; word end = next word's start (+500 ms for the last), so only starts
matter; `.` or `:` before the fraction, fraction optional; mixed plain and
word-timed lines are fine. LYS and TTML are the other word-timed formats (TTML
alone can upgrade a cached line-timed entry).

## Pipeline (`karaoke_lrc.py`)

Demucs two-stem vocal isolation → WhisperX transcription → forced alignment
(`WAV2VEC2_ASR_LARGE_LV60K_960H` for English) → lines → `.lrc` next to the audio.
With known lyrics, Whisper's words only locate each lyric line and the real
lyrics are timed instead (`known_lyrics.py`, below).

Setup: `uv sync`. Python 3.11 (whisperx needs 3.10–3.13), whisperx 3.8.6, demucs
4.1.0, torch 2.8.0 / torchaudio 2.8.0 / torchvision 0.23.0 from the cu128 index
via `[tool.uv.sources]` (PyPI's Windows wheels are CPU-only). Tested on Windows
10, RTX 3070 Ti 8 GB.

Things that matter, with the reason:
- **Phases, one model at a time**: all Demucs, then Whisper, unload, then align.
  Interleaving overflowed 8 GB (6 songs: 55 min → ~2 min); the large aligner next
  to Whisper made alignment 113 s instead of 6 s.
- `python -m demucs` via the running interpreter (a bare `demucs` needs an
  activated venv), `--shifts 0` so runs are reproducible, `--language en` by
  default (auto-detect hears only the first 30 s and once picked Norwegian).
- **Large aligner**: the base one crammed repeated lines into the first repeat
  (acoustic pop: mean error 0.39 → 0.11 s, >1 s off 7% → 1%). ~1 GB more VRAM,
  1.26 GB download.
- **Line breaking**: per-gap break scores (next word capitalised is the best cue,
  80% precision; Whisper segment; punctuation; silence, a weak cue because of
  held notes) and a whole-song optimal layout. Constants at the top of the file.
- **`retime_line_starts`**: a capitalised word with ≤0.3 s of voice before a
  pause, and unclaimed voice after it, moves to where the voice returns (pause =
  vocal stem 12 dB below its loud level for 0.25 s). Moved 16 words: 15 closer.
  Without the capital-letter condition it was a coin flip.
- **`fill_unheard`**: Whisper sometimes writes fewer repeats than are sung;
  re-transcribing a stretch of unclaimed voice (≥1.5 s after a pause) on its own
  recovers them. New words are kept only if half repeat a 4-word phrase from the
  song and they aren't mostly vocables: 24 added, all real (unfiltered: 26 of 80).
- Both fixes were tuned on the benchmark songs (no held-out set) and only run on
  the vocal stem, not with `--no-separate`.

Rejected: forcing the CTC path to the segment end (worse), Qwen3-ForcedAligner-
0.6B (sharper when right, but ~40% of words >1 s off on 2 of 20 songs; wav2vec2
closer on 108 vs 50 disagreements).

## Known lyrics (`known_lyrics.py`)

Whisper mishears sung words far more than it misses where the singing is: on six
metal songs, every lyric line it got wrong still had about the right number of
words in about the right place. So known lyrics are matched to Whisper's timed
words to find each line's stretch, then aligned there; each lyric line becomes an
LRC line. Lines Whisper heard none of share the gap between their neighbours and
are aligned together; lines sung more often than written are added by matching
Whisper's leftover words.

Lyrics come from `<stem>.lyrics.txt` next to the audio (or `--lyrics-dir`), then
LRCLIB (artist/title tags, duration ±3 s, retrying without a suffix like "(2017
Version)"), then Jamendo's API for its CC songs (needs a free client ID from
devportal.jamendo.com in `JAMENDO_CLIENT_ID`). A hit is saved as
`<stem>.lyrics.txt` to fix by hand and rerun with `--force`. `--offline` skips
the online lookups, `--no-lyrics` the whole feature. Sidecars are git-ignored.

Jamendo's API sometimes returns no results while reporting success, for any query
style, so the lookup tries four styles twice over (exact, artist catalogue,
free-text, title search; the title search ignores the artist filter, so results
are filtered on artist, title and duration). Since then it finds the same 21 of
22 songs every time.

## Benchmark (`eval_lrc.py`)

Ground truth: the 20 English JamendoLyrics songs
([Hugging Face](https://huggingface.co/datasets/jamendolyrics/jamendolyrics),
audio from `subsets/en/mp3/`), word timings in `test_songs/groundtruth/`.

The default report scores what a singer sees, without pairing .lrc words with
reference words: for each word when it's sung, is it highlighted then? "On time"
is 0.3 s early to 0.2 s late, after a karaoke listening study (Lizé Masclef,
Vaglio & Moussallam, "User-centered evaluation of lyrics-to-audio alignment",
ISMIR 2021: late lyrics are noticed sooner). "Bad lines" (under half the words
within 1 s) is the reliability number; "stray" counts .lrc words nobody sings
nearby. `--classic` gives the old pairing metrics (AAE, recall, break F1); they
mislead once the .lrc has a chorus more or less than the song, scoring whole
choruses against the wrong copy (Songwriterz: 6.9 s mean error, 98% of lines on
time), which once made known lyrics look worse than Whisper.

| 20 songs | Whisper's words | reference lyrics (best case) |
|---|---|---|
| on time | 75% | 85% |
| within 1 s | 81% | 95% |
| bad lines | 103 / 868 | 33 / 868 |
| stray words | 17% | 7% |
| songs with no bad lines | 6 | 9 |

With Jamendo's own lyrics (19 of the 20 have them) bad lines are 75 / 837. They
cut bad lines on most songs (Avercage 24 → 9, Ridgway 12 → 3) but wreck JASON
MILLER (2 → 26): Jamendo orders and repeats its sections differently from the
recording, so lines get pinned to the wrong stretch.

Tried and not adopted, all measured on this benchmark:
- Placement tweaks for out-of-order lyrics: capping a line's matched-word span,
  keeping Whisper's words where no lyric line fits, a looser repeat check.
  Together 75 → 47 bad lines on Jamendo lyrics, no change on reference lyrics,
  but mostly from one song, slightly more stray words, three more rules.
- Placing lines Whisper heard none of by splitting the gap by word count,
  dropping one-word matches, matching lines to sung phrases, or re-transcribing
  the gap with the lines as Whisper's prompt: none beat aligning the gap as one.
- Dropping lines too many to fit their gap; a stricter repeat check (the repeats
  it adds are mostly real).
- `--demucs-model htdemucs_ft` (the fine-tuned model; OpenKara ships the same
  Demucs as ONNX, so its splitter is no different): no better on Whisper's words
  (102 vs 103 bad lines, 75% on time both), worse on reference lyrics (46 vs 33,
  mostly Avercage 9 → 15 and LUNABLIND 1 → 5), and about twice as slow end to end.
- A scream detector (loud, weakly pitched vocal stem) to show syllables instead
  of words: it worked (72–81% of two fully harsh songs flagged, 0–3% of clean
  ones), but known lyrics already looked fine on the screamy songs checked.

## Test songs

All audio, generated `.lrc` files and lyric sidecars stay out of git (several
songs are CC "no derivatives"; commercial ones are bring-your-own).

- **JamendoLyrics** (20): the benchmark above.
- **Jamendo metal** (no word truth): Avenger Kills – Metal child / Rotten legion
  / The trap / Feeling my pain (Jamendo ids 1794820 / 1794821 / 1794825 /
  1794822), The Rinn – Into The Dark / Mirror (2017 Version) (1530457 /
  1530452). Download and tag with `uv run fetch_jamendo_audio.py`, from a list
  in `fetch_jamendo_audio.json` (copy the `.example.json`; git-ignored). All get
  lyrics from Jamendo; before known lyrics, Whisper got 83–95% of their words.
- **MUSDB18 metal** (hand-set word onsets,
  [Zenodo 15547046](https://zenodo.org/records/15547046)): Hollow Ground – Ill
  Fate, James Elder & Mark M Thompson – The English Actor, Timboz – Pony
  (screamed, no line times), We Fell From The Sky – Not You. `uv run
  musdb_truth.py` writes their ground truth (git-ignored: academic-use songs; line
  breaks from MUSDB-ALT). Audio needs a Zenodo access request, then `uv run
  musdb_extract.py` with `MUSDB18_PATH` set writes `test_songs/<name>.m4a`. The
  stems also carry clean vocals, which would show what Demucs costs.
- **Commercial metal** (bring your own copy, tag artist/title for LRCLIB): Iron
  Maiden – Aces High, King Gizzard – Self-Immolate, System Of A Down – B.Y.O.B.,
  plus harsher Slipknot – Duality, Arch Enemy – Nemesis, Lamb of God – Redneck,
  Killswitch Engage – My Curse. Use studio cuts: the ±3 s duration match skips
  live versions and edits. `fetch_youtube_audio.py` downloads a list from
  `fetch_youtube_audio.json` (copy the `.example.json`; git-ignored).
- **Public-domain 78s**: Over There (1917), Shine On, Harvest Moon (1909). US
  rules as of 2026: compositions published 1930 or earlier *and* recordings 1925
  or earlier. Sources: Internet Archive Great 78 Project, Library of Congress
  National Jukebox.

## Known rough edges

- WhisperX drops timings for some tokens (numerals, odd glyphs); those words are
  skipped.
- Pre-1925 recordings: Demucs still won on an unrestored 1909 78 (177 vs 155
  words; `--no-separate` hallucinated a "You." in the silent intro).
- Repeats are mostly handled but need a pause before the line and a phrase that
  repeats elsewhere; a legato line with no breath or a dropped one-off isn't
  caught.
- Capitalised proper nouns can start a line early ("since / April, January").
- Without known lyrics, budget for hand-editing: fixing a word's text doesn't
  touch its timing.

## Next steps

1. Import into OpenKara and check that word highlighting tracks the music.
2. Tune line breaking against what reads well on screen.
3. Real timing numbers for metal once the MUSDB18 audio is in.
4. Only then consider an OpenKara "generate lyrics" button that runs this script.
