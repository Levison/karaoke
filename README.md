# karaoke

Turns an audio file into a word-timed enhanced `.lrc` for
[OpenKara](https://github.com/thedavidweng/OpenKara). See
[OPENKARA_NOTES.md](OPENKARA_NOTES.md) for setup, usage and benchmarks.

## Disclaimer

This project is a tool. We don't post, host or distribute songs or lyrics for
public use. Lyrics you generate with it come from audio you supply; you are
responsible for having the rights to that audio and to any lyrics you share.

The only song material in this repo is `test_songs/groundtruth/`, kept to
measure timing accuracy: reference lyrics and word timings for 20 English songs
from the
[JamendoLyrics MultiLang dataset](https://huggingface.co/datasets/jamendolyrics/jamendolyrics).
Each song is under its artist's Creative Commons license (BY, BY-SA, BY-ND,
BY-NC, BY-NC-SA or BY-NC-ND; see the dataset's `metadata.jsonl` for which is
which), and copyright stays with the artists.

Audio files and generated `.lrc` files are not committed (see `.gitignore`).
Several of these songs are "no derivatives", and a generated `.lrc` is derived
from the song.

## License

The code in this repo is under the [MIT License](LICENSE). The license does not
cover `test_songs/groundtruth/`, which keeps the licenses above.
