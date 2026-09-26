# OpenKara scoping notes — karaoke lyrics pipeline

Findings from reading `thedavidweng/OpenKara` @ main (Apache-2.0, Tauri 2 + Rust + React).
Source: `src-tauri/src/lyrics/` — `fetch.rs`, `parser.rs`, `acquisition.rs`.

## The big one: no fork needed

OpenKara's lyrics acquisition chain is:

```
cache → embedded tags → TTML/LYS/LRC sidecar → AMLL → LRCLIB → LrcApi
```

Cached lyrics come first, and an imported `.lrc` is cached, so it always wins
over LRCLIB. Public-domain songs that LRCLIB has never heard of work fine. No
code change to OpenKara required.

## Getting a `.lrc` into OpenKara (checked against build 74240930, 2026-09-19)

The sidecar step in the chain does **not** see our files. Import copies the audio
to `<library>/media/<sha256>.<ext>` (`import/ingest.rs`), and
`fetch.rs::read_sidecar_lyrics` looks for `<sha256>.lrc` in that folder.

What works instead is importing the `.lrc` in the same import as the audio. The
frontend (`runtime/import-workflow.ts`) imports the audio first, then passes the
`.lrc` files to `import_lyrics_files` (`commands/lyrics.rs`), which caches each
one as `Manual` lyrics for the song it matches:
- File-stem match: never hits, because the song's stem is its hash.
- Fallback: the `.lrc`'s `[ar:]` and `[ti:]` must both equal the song's artist
  and title (case-insensitive). The title falls back to the audio file name when
  there's no title tag. An empty `[ar:]` matches a song with no artist tag, but a
  missing `[ar:]` matches nothing.
- `karaoke_lrc.py` writes both tags from the audio's own tags (mutagen,
  `fix_lrc_tags.read_tags`). Use `uv run fix_lrc_tags.py` on older `.lrc` files.
  On 2026-09-18 a re-run of the old generator silently undid the tags.
- Brackets in the title or artist (e.g. *Vision [Radio Edit]*) truncate at the
  first `]` upstream, so those files match nothing. Fixed in our checkout on
  branch `fix/lyrics-bracketed-metadata-tags` (upstream issue #465, PR #466);
  verified 2026-09-22, all 23 files matched. On a build without the fix, use
  Edit lyrics in OpenKara for those.
- An unmatched `.lrc` shows an error toast.

Sidecar rules, for reference: same folder and stem as the (library) audio file,
case-insensitive extension, `.ttml` > `.lys` > `.lrc`, and an unparseable file
is silently skipped.

Keep the library folder outside the song folders. Folder import scans 3 levels
deep, so re-importing a folder that contains the library also picks up its
`media/` copies.

## Enhanced LRC is supported → word-by-word highlighting for free

`parser.rs::parse_word_tokens` handles inline `<mm:ss.xx>` word tags. Its own
test fixture:

```
[00:12.00]<00:12.00>I <00:12.30>see <00:12.60>trees
```

Rules worth knowing:
- Line-level `[mm:ss.xx]` tag is still required at line start.
- Word `end_ms` is inferred: next word's start, or +500ms for the last word.
  So you only need per-word **start** times.
- Timestamps accept `.` or `:` as the fractional separator, and the fractional
  part is optional.
- Mixed plain and word-timed lines in one file are fine.

`LYS` is the other word-timed format: `[0]Hello(1000,500) World(1500,500)`
(absolute ms, start + duration). TTML is the richest — it's what AMLL serves and
it's the only source allowed to word-time-upgrade a cached line-timed entry.
LRC is the path of least resistance and gets you the same karaoke fill.

## `karaoke_lrc.py`

Generates enhanced LRC. Pipeline: Demucs two-stem vocal isolation → WhisperX
transcribe + forced alignment → grouped lines → `.lrc` next to the audio.

Verified from here: timestamp formatting, line grouping, and output shape round-trip
against a reimplementation of OpenKara's parser rules.

Line-breaking constants are at the top of the file (`TARGET_LINE_CHARS`,
`LINE_COST`, `BREAK_*` weights, …). Higher `LINE_COST` = fewer, longer lines.

## First real run — 2026-09-18 (Windows 10, RTX 3070 Ti 8 GB, Python 3.11)

Setup: `uv sync`. `pyproject.toml` + `uv.lock` pin the working set: Python 3.11
(whisperx needs 3.10–3.13), whisperx 3.8.6, demucs 4.1.0, and torch 2.8.0 /
torchaudio 2.8.0 / torchvision 0.23.0 from the **cu128** index. PyPI's Windows
torch wheels are CPU-only, so those three are routed there via
`[tool.uv.sources]`. torchvision is only a whisperx dependency, but it's listed
directly so the source applies. Run with `uv run karaoke_lrc.py …`.
(Originally set up with pip; moved to uv the same day.)

Bugs found and fixed in `karaoke_lrc.py`:
- Demucs was called as `demucs` on PATH → silently skipped unless the venv was
  activated. Now `python -m demucs` via the running interpreter.
- Language auto-detect listens to the first 30 s — often an instrumental intro.
  It picked Norwegian on one song, pulled a 3.6 GB alignment model and aligned
  English with it. Default is now `--language en`; `--language auto` still exists.
- Interleaving Demucs (child process) with WhisperX (~4.5 GB VRAM) overflowed
  8 GB into shared memory: 6 songs took 55 min. Now all separation runs first,
  then WhisperX loads once — same 6 songs in ~2 min.
- Demucs' default single random shift made every run's output differ; now
  `--shifts 0`, so the same input always gives the same `.lrc`.
- UTF-8 output on Windows; "Mr." no longer ends a line.

Line breaking rewritten: per-gap break scores (next word capitalised — Whisper
capitalises lyric-line starts, and it's the best single cue at 80% precision;
segment boundary; punctuation; silence) + whole-song optimal layout instead of
greedy rules. Pauses turned out to be a weak cue (held notes mid-line).

Benchmark (first pass): 3 English songs from [JamendoLyrics](https://huggingface.co/datasets/jamendolyrics/jamendolyrics)
(word-level ground truth) + 3 public-domain 78s in `test_songs/`. Later expanded
to all 20 English songs, see "Repeated lines" below. Score with
`uv run eval_lrc.py test_songs\*.lrc`.

| song | word recall | median onset error | mean onset error | line-break F1 |
|---|---|---|---|---|
| hip-hop | 88% | 0.05 s | 0.06 s | 82% |
| acoustic pop | 93% | 0.06 s | 0.11 s | 78% |
| rock, non-native singer | 45% | 0.15 s | 0.69 s | 72% |

Line-break F1 before the rewrite was 66% / 42% / 26%.

### Larger alignment model (same day)

WhisperX's English aligner (`WAV2VEC2_ASR_BASE_960H`) crammed repeated lines
into the first repeat. In the acoustic-pop song, the third chorus's four "I want
you to feel" repeats all started within 4 s, with the last two 5–6 s early. The
cause is that the aligner confuses identical repeats, so choosing when to break
lines doesn't help. Switching to `WAV2VEC2_ASR_LARGE_LV60K_960H` (now the
English default, `--align-model` to override):

| song | mean onset error | onsets >1 s off | line-break F1 |
|---|---|---|---|
| acoustic pop | 0.39 → 0.11 s | 7% → 1% | 82% → 78% |
| rock | 0.71 → 0.69 s | 11% → 11% | 72% (same) |
| hip-hop | 0.06 s (same) | 0% (same) | 82% (same) |

The F1 drop is a side effect of better timing. A leftover early "I" now leaves a
gap before "want", and the line breaker splits there ("…feel I / want you to feel").

Tried and rejected: forcing the CTC path to run to the end of Whisper's segment,
instead of stopping where the transcript's score peaks. It made the base model
worse (mean error 0.43 s) and undid half the large model's gain (0.32 s).

The large model needs ~1 GB more VRAM than base. Alongside Whisper that overflowed
8 GB, and alignment took 113 s instead of 6 s. So `karaoke_lrc.py` now
transcribes every file, unloads Whisper (in-process, which does free the VRAM),
then aligns. All 6 songs: 83 s. One-time download: 1.26 GB.

### Repeated lines: bigger benchmark, Qwen3, two fixes (same evening)

**Benchmark.** `test_songs/` now holds all 20 English JamendoLyrics songs with
ground truth (download audio from `subsets/en/mp3/`; `mp3/` holds symlinks).
The 3-song set had only one song with repeated choruses. 20-song means:

| pipeline | mean onset error | median | within 0.3 s | >1 s off | line-break F1 |
|---|---|---|---|---|---|
| large aligner | 0.78 s | 0.07 s | 91% | 5% | 78% |
| + line-start fix + dropped-repeat fill (current) | 0.73 s | 0.07 s | 91% | 4% | 79% |

Mean error is dominated by a few songs (The Rinn 9 s, Rxbyn 0.9 s); the median
and ">1 s off" are steadier. Not yet checked whether those outliers are real
misalignments or `eval_lrc.py` pairing a word with the wrong repeat.

**The leftover "early repeats" were one word.** In all four cases the only
mistimed word was the "I" after a held "feel". The aligner put it inside the
held vowel, 1.1–1.8 s early, and the next word was within 0.1 s. It wasn't
repeat confusion.

**Qwen3-ForcedAligner-0.6B: tried, rejected.** It's a new (2026) non-CTC aligner
reporting ~3× better timing than WhisperX on speech. On these 20 songs it's
sharper when right (median error lower on 14 of 20) but drifts badly on others
(~40% of words >1 s off on 2 songs):

| Qwen3 variant | within 0.3 s | >1 s off |
|---|---|---|
| per Whisper segment | 83% | 10% |
| per segment, ±1 s padding | 83% | 11% |
| on the original mix | 71% | 19% |
| whole song at once | 56% | 38% |

As a second opinion it's no better. Where it disagrees with wav2vec2 on a single
word, wav2vec2 is closer to the truth 108 times vs Qwen's 50. Weights: 1.75 GB.

**Fix 1: `retime_line_starts`.** A capitalised word that has ≤0.3 s of voice
before a pause, while ≥0.1 s of voice after the pause belongs to no word, gets
moved to where the voice returns. Pause = vocal stem 12 dB below the song's loud
level for 0.25 s (Demucs bleed means breaths are dips, not silence). It moved 16
words that have ground truth: 15 closer, 1 further. It caught all four Cortez
"I"s, which also fixed the "…feel I / want you to feel" line splits (acoustic
pop F1 78% → 83%). Without the capital-letter condition it was a coin flip
(18 closer, 28 further): line-*ending* words are also short and followed by a
breath.

**Fix 2: `fill_unheard`, snip and retranscribe.** Whisper sometimes writes fewer
repeats than are sung. Transcribing the same stretch on its own recovers them. In
the acoustic-pop song's chorus 1, the full run got 2 of 4 "I want you to feel"
and the snip got all 4. The pipeline looks for voice that comes back after a
pause and runs ≥1.5 s before the next word, transcribes that stretch alone, and
keeps the words only if ≥ half of them repeat a 4-word phrase from elsewhere in
the song and they aren't mostly vocables. Unfiltered, only 26 of 80 added words
were real lyrics (invented lines like "I'm a falcon!", plus "do do do" / "ah ah"
fills the ground truth doesn't count). Filtered: 24 added, 24 real (acoustic pop
+10, Rxbyn +14). Costs one extra Whisper load, only when a song has such
stretches (~17 s on the 23 test songs).

Caveat for both fixes: the thresholds and filters were chosen by looking at
these same 20 songs, and there's no held-out set yet. Both fixes only run on the
vocal stem. They're skipped with `--no-separate`, where the band fills the pauses.

### Heavy metal set (2026-09-25)

JamendoLyrics has only one metal song in English (The Rinn, *Voices*, the 9 s
outlier), so six more English metal tracks from Jamendo were added to
`test_songs/`. They have no word timings, so `eval_lrc.py` skips them. Judge
them by eye against the lyrics on each track's Jamendo `/lyrics` page. Download
with `curl -L -o <name>.mp3 https://prod-1.storage.jamendo.com/download/track/<id>/mp32/`.

| file stem | Jamendo id | style |
|---|---|---|
| `Avenger_Kills_-_Metal_child` | 1794820 | 80s heavy/power metal, male vocals |
| `Avenger_Kills_-_Rotten_legion` | 1794821 | 〃 |
| `Avenger_Kills_-_The_trap` | 1794825 | 〃 |
| `Avenger_Kills_-_Feeling_my_pain` | 1794822 | 〃 |
| `The_Rinn_-_Into_The_Dark` | 1530457 | symphonic power metal, female vocals, same album as *Voices* |
| `The_Rinn_-_Mirror__2017_Version_` | 1530452 | 〃 |

Lyrics page: `https://www.jamendo.com/track/<id>/x/lyrics`. Avenger Kills'
lyrics read as written by non-native speakers, so expect odd grammar there that
isn't Whisper's fault.

First run (current pipeline, all defaults). No timing truth, so these are text
coverage against the posted lyrics only. "Lines heard" = lyric lines with ≥60%
of their words found in order in the .lrc. "In lyrics" = .lrc words that occur
anywhere in the posted lyrics, a rough inverse of mishearings and invented words.

| song | lines heard | words heard | .lrc words in lyrics |
|---|---|---|---|
| Avenger Kills – Feeling my pain | 15/18 | 89% | 88% |
| Avenger Kills – Metal child | 18/20 | 88% | 89% |
| Avenger Kills – Rotten legion | 19/20 | 84% | 84% |
| Avenger Kills – The trap | 19/20 | 90% | 92% |
| The Rinn – Into The Dark | 23/28 | 83% | 82% |
| The Rinn – Mirror | 31/31 | 95% | 94% |

That's in line with the first JamendoLyrics benchmark (word recall 88% hip-hop,
93% acoustic pop, 45% rock with a non-native singer), though that recall comes
from `eval_lrc.py`, a different measure. Whisper writing a ~400-letter "Yoooo…" for a held scream in
*The trap* was the only oddity: the aligner rejected it and no garbage reached
the .lrc.

**Metal with real word timings: MUSDB18.** Four MUSDB18 test songs are tagged
Heavy Metal, and they have hand-set word onsets
([Zenodo 15547046](https://zenodo.org/records/15547046)).
`uv run musdb_truth.py` writes their ground truth into `test_songs/groundtruth/`
under the MUSDB name (e.g. `Timboz - Pony.words.csv`). These files are
git-ignored, since MUSDB18's songs are academic-use only. Line breaks come from
MUSDB-ALT's line times. Timboz – Pony has none (screamed vocals), so its break
columns show "-".

| song | words | lines |
|---|---|---|
| Hollow Ground – Ill Fate | 145 | 30 |
| James Elder & Mark M Thompson – The English Actor | 213 | 41 |
| Timboz – Pony | 141 | – |
| We Fell From The Sky – Not You | 258 | 43 |

The audio isn't included. Request MUSDB18 on Zenodo (academic use only), then
extract each mix as `test_songs/<name>.m4a` (the command is in the script's
docstring). The `.stem.mp4` also carries the clean vocals (stream 4), which
would show how much Demucs costs.

### Known lyrics (2026-09-26)

On the six metal songs, every lyric line the pipeline missed still had about
the right number of words in about the right place. Whisper was mishearing
lines, not missing them. So when the lyrics are known, Whisper's timed words
now only locate each lyric line, and the aligner times the real lyric words
(`known_lyrics.py`). Each lyric line becomes an LRC line.

Where lyrics come from, in order: `<stem>.lyrics.txt` next to the audio (or
`--lyrics-dir`), then LRCLIB by artist/title tags and duration (±3 s; a title
suffix like "(2017 Version)" is dropped if the full title finds nothing). An
LRCLIB hit is saved as `<stem>.lyrics.txt`, so a wrong word can be fixed there
and the song rerun with `--force`. Sidecars are git-ignored. `--no-lrclib`
disables the lookup; `--no-lyrics` restores the old behaviour.

20 JamendoLyrics songs, with their reference lyrics as the known lyrics (best
case), against Whisper's words:

| | Whisper's words | known lyrics |
|---|---|---|
| word recall | 81% | 100% |
| line-break F1 | 79% | 100% |
| onset error, words both runs have (mean / median) | 0.48 / 0.06 s | 0.49 / 0.06 s |
| >1 s off, all words | 3.7% | 7.9% |

Text and lines become exact, and words Whisper heard keep the same timing. The
new words, the 19% Whisper never heard, are the weak spot. For lines with no
word matching Whisper's, about half the words are over 1 s off, because
there's nothing to pin them to. Tried and rejected, none better: aligning the
whole gap as one text (kept, as it's simplest), dropping one-word "anchors",
matching lines to the vocal stem's phrases by length, and re-transcribing the
gap with the missing lines as Whisper's prompt (Whisper then invents them where
they aren't sung).

Real LRCLIB lyrics, one song: *The Rinn – Voices* (metal, the 9 s outlier) went
from 57% recall and 9.04 s mean error to 96% and 0.11 s, 95% of words within
0.3 s. The reference lyrics scored far worse on the same song (9.10 s); not yet
checked why.

Unwritten repeats (a chorus the lyrics write once) are found by matching
Whisper's leftover words against lyric lines. With reference lyrics, which
write every repeat, it still added 1–31 lines per song. Turning it off made
timing slightly worse, so it's kept, but it's a false-positive source.

## Known rough edges to expect

- WhisperX drops timings for some tokens (numerals, odd glyphs); those words are
  skipped rather than mistimed.
- Pre-1925 recordings: heavy surface noise, often mono. Demucs was trained on
  modern stereo mixes and will struggle. Try `--no-separate` and compare — on very
  noisy sources the original mix sometimes transcribes better than a mangled stem.
  *Tested:* on an unrestored 1909 78 (*Shine On, Harvest Moon*) Demucs still won
  (177 vs 155 words), and `--no-separate` hallucinated a "You." in the silent intro.
  Noise mostly costs Whisper's punctuation, not its words.
- Repeated phrases: mostly handled now (large aligner, `retime_line_starts`,
  `fill_unheard`), but still check choruses when hand-editing. The fixes need a
  pause before the line and a phrase that repeats elsewhere in the song. A
  legato line with no breath, or a dropped line sung only once, isn't caught.
- Capitalised proper nouns can still start a line early ("since / April, January").
- Whisper mishears sung lyrics more than speech. Budget for hand-editing the
  `.lrc`. Since word `end` times are inferred, you can fix a word's text without
  touching any timing.

## Public domain, US, as of 2026

- Compositions: published 1930 or earlier.
- Sound recordings: 1925 or earlier (they follow a separate, stricter clock).
- **Both** must clear — a modern recording of an old song is still protected.
- Sources: Internet Archive Great 78 Project, Library of Congress National Jukebox.

## Next steps

1. ~~Set up a venv, install, confirm ffmpeg on PATH.~~ Done — see "First real run".
2. ~~Run on 2–3 test songs.~~ Done on 23. The `.lrc` files are generated into `test_songs/` but not committed.
3. Import into OpenKara, confirm word highlighting tracks the instrumental.
4. Tune the line-breaking constants against what actually reads well on screen.
   (Scored version is in; check it on screen in OpenKara before tuning further.)
5. Only then consider forking OpenKara to add a "generate lyrics" button that
   shells out to this script.
