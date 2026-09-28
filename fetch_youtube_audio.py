#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["yt-dlp>=2025.1", "mutagen>=1.47"]
# ///
"""
fetch_youtube_audio.py — search YouTube and save audio into test_songs/ for local testing.

Needs ffmpeg on PATH (yt-dlp uses it to encode MP3). You are responsible for only
downloading material you have the right to use.

Usage:
    cp fetch_youtube_audio.example.json fetch_youtube_audio.json   # edit tracks; file is git-ignored
    uv run fetch_youtube_audio.py --file fetch_youtube_audio.json
    uv run fetch_youtube_audio.py --stem Artist_-_Song --artist Artist --title Song --duration 240 "search words"
    uv run fetch_youtube_audio.py --list "search words"
    uv run fetch_youtube_audio.py -o test_songs/foo.mp3 "https://www.youtube.com/watch?v=..."
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import mutagen

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "test_songs"
DEFAULT_LIST = ROOT / "fetch_youtube_audio.json"

DURATION_SLACK = 3.0  # same as known_lyrics.py / LRCLIB matching


@dataclass(frozen=True)
class TrackSpec:
    stem: str
    artist: str
    title: str
    duration: float | None
    query: str


def load_track_list(path: Path) -> tuple[Path, list[TrackSpec]]:
    """Read a JSON batch list. out_dir in the file is relative to the list's directory."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object at the top level")
    raw_out = data.get("out_dir", "test_songs")
    out_dir = Path(raw_out)
    if not out_dir.is_absolute():
        out_dir = (path.parent / out_dir).resolve()
    tracks_raw = data.get("tracks")
    if not isinstance(tracks_raw, list) or not tracks_raw:
        raise SystemExit(f"{path}: need a non-empty \"tracks\" array")
    specs: list[TrackSpec] = []
    for i, row in enumerate(tracks_raw, start=1):
        if not isinstance(row, dict):
            raise SystemExit(f"{path}: tracks[{i - 1}] must be an object")
        stem = row.get("stem")
        query = row.get("query") or row.get("url")
        if not stem or not query:
            raise SystemExit(f"{path}: tracks[{i - 1}] needs \"stem\" and \"query\" (or \"url\")")
        duration = row.get("duration")
        if duration is not None and not isinstance(duration, (int, float)):
            raise SystemExit(f"{path}: tracks[{i - 1}].duration must be a number")
        specs.append(
            TrackSpec(
                str(stem),
                str(row.get("artist") or ""),
                str(row.get("title") or stem),
                float(duration) if duration is not None else None,
                str(query),
            )
        )
    return out_dir, specs


def yt_dlp_base() -> list[str]:
    return [sys.executable, "-m", "yt_dlp"]


def is_url(text: str) -> bool:
    return bool(re.match(r"https?://", text.strip()))


def search_entries(query: str, limit: int) -> list[dict]:
    url = f"ytsearch{limit}:{query}"
    cmd = [
        *yt_dlp_base(),
        "--flat-playlist",
        "--dump-single-json",
        "--no-warnings",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout.strip() or "yt-dlp search failed")
    data = json.loads(proc.stdout)
    entries = data.get("entries") or []
    return [e for e in entries if e]


def choose_pick(entries: list[dict], duration: float | None, explicit_pick: int | None) -> int:
    if explicit_pick is not None:
        return explicit_pick
    if duration is not None:
        for i, entry in enumerate(entries, start=1):
            d = entry.get("duration")
            if isinstance(d, (int, float)) and abs(d - duration) <= DURATION_SLACK:
                return i
    return 1


def format_entry(entry: dict, index: int) -> str:
    title = entry.get("title") or "?"
    vid = entry.get("id") or entry.get("url") or "?"
    duration = entry.get("duration")
    dur = f"{duration:.0f}s" if isinstance(duration, (int, float)) else "?s"
    uploader = entry.get("uploader") or entry.get("channel") or "?"
    return f"  {index}. [{dur}] {title} — {uploader}  (id {vid})"


def match_filter(duration: float | None) -> str | None:
    if duration is None:
        return None
    lo = max(0, duration - DURATION_SLACK)
    hi = duration + DURATION_SLACK
    return f"duration >= {lo} & duration <= {hi}"


def download(
    source: str,
    out_path: Path,
    *,
    duration: float | None,
    dry_run: bool,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    template = str(out_path.with_suffix(".%(ext)s"))
    cmd = [
        *yt_dlp_base(),
        "-x",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "--no-playlist",
        "-o",
        template,
        "--print",
        "after_move:filepath",
    ]
    filt = match_filter(duration)
    if filt:
        cmd.extend(["--match-filter", filt])
    if dry_run:
        cmd.append("--simulate")
    cmd.append(source)

    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()
        raise SystemExit(msg or "yt-dlp download failed")

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if dry_run:
        return out_path.with_suffix(".mp3")
    if not lines:
        raise SystemExit("yt-dlp did not print an output path")
    return Path(lines[-1])


def write_tags(path: Path, artist: str, title: str) -> None:
    audio = mutagen.File(path, easy=True)
    if audio is None:
        raise SystemExit(f"Could not open tags on {path}")
    audio["artist"] = artist
    audio["title"] = title
    audio.save()


def resolve_output(out_dir: Path, stem: str | None, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if not stem:
        raise SystemExit("Pass --stem (or --out) so the file name matches test_songs conventions")
    return out_dir / stem


def run_track(
    spec: TrackSpec,
    out_dir: Path,
    *,
    search_limit: int,
    pick: int | None,
    dry_run: bool,
    list_only: bool,
) -> None:
    print(f"\n{spec.artist} — {spec.title}")
    if list_only:
        for i, entry in enumerate(search_entries(spec.query, search_limit), start=1):
            print(format_entry(entry, i))
        return

    if is_url(spec.query):
        source = spec.query
    else:
        entries = search_entries(spec.query, search_limit)
        if not entries:
            raise SystemExit(f"No YouTube results for: {spec.query}")
        chosen_index = choose_pick(entries, spec.duration, pick)
        if chosen_index < 1 or chosen_index > len(entries):
            raise SystemExit(f"--pick {chosen_index} out of range (1–{len(entries)})")
        chosen = entries[chosen_index - 1]
        print(format_entry(chosen, chosen_index))
        vid = chosen.get("id")
        if not vid:
            raise SystemExit("Search result had no video id")
        source = f"https://www.youtube.com/watch?v={vid}"

    out = resolve_output(out_dir, spec.stem, None)
    final = download(source, out.with_suffix(".mp3"), duration=spec.duration, dry_run=dry_run)
    if dry_run:
        print(f"  (dry run) would write {final}")
        return
    if final != out.with_suffix(".mp3") and final.is_file():
        target = out.with_suffix(".mp3")
        if target.is_file():
            target.unlink()
        final.rename(target)
        final = target
    write_tags(final, spec.artist, spec.title)
    print(f"  -> {final}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "query",
        nargs="*",
        help="YouTube URL or search words (single-track mode; omit when using --file)",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help=f"JSON track list to download (default if omitted: {DEFAULT_LIST.name}; copy from .example.json)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help=f"output folder (default: {DEFAULT_OUT.name}/)")
    parser.add_argument("--stem", help="output file name without extension (e.g. Iron_Maiden_-_Aces_High)")
    parser.add_argument("--artist", help="ID3 artist tag for LRCLIB lookup")
    parser.add_argument("--title", help="ID3 title tag for LRCLIB lookup")
    parser.add_argument(
        "--duration",
        type=float,
        help=f"expected length in seconds; yt-dlp skips results outside ±{DURATION_SLACK:.0f}s",
    )
    parser.add_argument(
        "--search-results",
        type=int,
        default=5,
        metavar="N",
        help="how many YouTube search hits to consider (default: 5)",
    )
    parser.add_argument(
        "--pick",
        type=int,
        default=None,
        metavar="N",
        help="which search hit to download (default: first whose length matches --duration, else 1)",
    )
    parser.add_argument("--list", action="store_true", dest="list_only", help="print search hits and exit")
    parser.add_argument("--dry-run", action="store_true", help="show yt-dlp command without writing files")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="exact output .mp3 path (overrides --stem; for single URL downloads)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    list_path = args.file
    if list_path is None and not args.query:
        list_path = DEFAULT_LIST

    if list_path is not None:
        if args.query:
            print("Use either --file or a search query/URL, not both.", file=sys.stderr)
            return 2
        if not list_path.is_file():
            example = ROOT / "fetch_youtube_audio.example.json"
            hint = f"Copy {example.name} to {list_path.name} and edit the tracks you want."
            raise SystemExit(f"No list file at {list_path}. {hint}")
        out_dir, specs = load_track_list(list_path)
        if args.out_dir != DEFAULT_OUT:
            out_dir = args.out_dir
        for spec in specs:
            run_track(
                spec,
                out_dir,
                search_limit=args.search_results,
                pick=args.pick,
                dry_run=args.dry_run,
                list_only=args.list_only,
            )
        return 0

    if not args.query:
        print("Give a search query, URL, or --file.", file=sys.stderr)
        return 2

    text = " ".join(args.query)
    stem = args.stem or (args.out.stem if args.out else None)
    if not args.list_only and not stem and not args.out:
        print("Pass --stem or -o for a single download.", file=sys.stderr)
        return 2

    spec = TrackSpec(
        stem=stem or "download",
        artist=args.artist or "",
        title=args.title or stem or "download",
        duration=args.duration,
        query=text,
    )
    if args.list_only:
        for i, entry in enumerate(search_entries(text, args.search_results), start=1):
            print(format_entry(entry, i))
        return 0

    out = args.out or resolve_output(args.out_dir, stem, None)
    if is_url(text):
        source = text
    else:
        entries = search_entries(text, args.search_results)
        if not entries:
            raise SystemExit(f"No YouTube results for: {text}")
        chosen_index = choose_pick(entries, args.duration, args.pick)
        if chosen_index < 1 or chosen_index > len(entries):
            raise SystemExit(f"--pick {chosen_index} out of range (1–{len(entries)})")
        chosen = entries[chosen_index - 1]
        print(format_entry(chosen, chosen_index))
        vid = chosen.get("id")
        if not vid:
            raise SystemExit("Search result had no video id")
        source = f"https://www.youtube.com/watch?v={vid}"

    final = download(source, out.with_suffix(".mp3"), duration=args.duration, dry_run=args.dry_run)
    if args.dry_run:
        print(f"(dry run) would write {final}")
        return 0
    if final != out.with_suffix(".mp3") and final.is_file():
        target = out.with_suffix(".mp3")
        if target.is_file():
            target.unlink()
        final.rename(target)
        final = target
    if args.artist or args.title:
        write_tags(final, args.artist or "", args.title or final.stem)
    print(f"-> {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
