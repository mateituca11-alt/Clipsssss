"""Clip MVP: long video URL -> loudest moments -> 9:16 captioned clips.

Usage: python clipper.py <url> [--clips 8] [--len 60] [--gap 240]
"""
import argparse
import pathlib
import subprocess

import numpy as np
from faster_whisper import WhisperModel

OUT = pathlib.Path("out")

# Everything in one file: this is the GitHub Actions workflow (also needs no
# requirements.txt, packages are installed inline). Run `python clipper.py --setup`
# to write it to .github/workflows/clip.yml, or copy it there by hand.
WORKFLOW = """name: clip

on:
  workflow_dispatch:
    inputs:
      url:
        description: Video URL (Twitch VOD or YouTube)
        required: true
      clips:
        description: Number of clips
        default: "8"

jobs:
  run:
    runs-on: ubuntu-latest
    timeout-minutes: 300
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: sudo apt-get update && sudo apt-get install -y ffmpeg
      - run: pip install yt-dlp faster-whisper numpy
      - name: Make clips
        env:
          URL: ${{ inputs.url }}
          CLIPS: ${{ inputs.clips }}
        run: python clipper.py "$URL" --clips "$CLIPS"
      - uses: actions/upload-artifact@v4
        with:
          name: clips
          path: out/clip_*.mp4
"""


def setup_workflow():
    p = pathlib.Path(".github/workflows/clip.yml")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(WORKFLOW, encoding="utf-8")
    print(f"wrote {p}")


def run(cmd, cwd=None):
    subprocess.run(cmd, check=True, cwd=cwd)


def download_audio(url):
    OUT.mkdir(exist_ok=True)
    run(["yt-dlp", "-f", "bestaudio", "-x", "--audio-format", "m4a",
         "-o", str(OUT / "audio.%(ext)s"), url])
    return OUT / "audio.m4a"


def loudness_per_second(path):
    """Stream audio through ffmpeg, return RMS loudness for each second."""
    p = subprocess.Popen(
        ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "s16le",
         "-ac", "1", "-ar", "8000", "-"],
        stdout=subprocess.PIPE)
    size = 8000 * 2
    vals = []
    while True:
        b = p.stdout.read(size)
        if len(b) < size:
            break
        a = np.frombuffer(b, dtype=np.int16).astype(np.float32)
        vals.append(float(np.sqrt((a * a).mean())))
    return np.array(vals)


def pick_moments(vals, n, clip_len, min_gap):
    """Smooth loudness, take the top peaks that are at least min_gap apart."""
    k = 15
    sm = np.convolve(vals, np.ones(k) / k, mode="same")
    z = (sm - sm.mean()) / (sm.std() + 1e-6)
    chosen = []
    for i in np.argsort(-z):
        if all(abs(int(i) - c) >= min_gap for c in chosen):
            chosen.append(int(i))
        if len(chosen) == n:
            break
    moments = []
    for c in sorted(chosen):
        start = max(0, c - int(clip_len * 0.65))  # build-up before the peak
        moments.append((start, start + clip_len, float(z[c])))
    return moments


def download_section(url, start, end, dest):
    run(["yt-dlp", "-f", "bv*[height<=720]+ba/b[height<=720]",
         "--download-sections", f"*{start}-{end}",
         "--force-keyframes-at-cuts", "--merge-output-format", "mp4",
         "-o", str(dest), url])


def ts(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{int((s % 1) * 1000):03}"


def write_srt(model, video, srt_path, words_per_line=4):
    segs, _ = model.transcribe(str(video), word_timestamps=True)
    words = [w for s in segs for w in s.words]
    lines = []
    for i in range(0, len(words), words_per_line):
        grp = words[i:i + words_per_line]
        text = " ".join(w.word.strip() for w in grp).upper()
        lines.append(f"{len(lines) + 1}\n{ts(grp[0].start)} --> {ts(grp[-1].end)}\n{text}\n")
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def render(raw, srt, final):
    style = ("FontSize=18,Bold=1,Outline=2,Shadow=0,Alignment=2,MarginV=90,"
             "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000")
    vf = f"crop=ih*9/16:ih,scale=1080:1920,subtitles={srt.name}:force_style='{style}'"
    run(["ffmpeg", "-y", "-v", "error", "-i", raw.name, "-vf", vf,
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-c:a", "aac", final.name], cwd=OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--len", type=int, default=60)
    ap.add_argument("--gap", type=int, default=240, help="min seconds between clips")
    ap.add_argument("--setup", action="store_true", help="write the GitHub workflow file")
    a = ap.parse_args()
    if a.setup:
        return setup_workflow()
    if not a.url:
        ap.error("give a video URL (or use --setup)")

    print("1/4 downloading audio")
    audio = download_audio(a.url)
    print("2/4 finding loud moments")
    vals = loudness_per_second(audio)
    moments = pick_moments(vals, a.clips, a.len, a.gap)
    print(f"   picked {len(moments)} moments from {len(vals) / 3600:.1f}h")

    model = WhisperModel("base", device="cpu", compute_type="int8")
    for i, (s, e, score) in enumerate(moments, 1):
        print(f"3/4 clip {i}/{len(moments)} ({s}s-{e}s, score {score:.1f})")
        raw = OUT / f"raw_{i}.mp4"
        srt = OUT / f"clip_{i}.srt"
        final = OUT / f"clip_{i}.mp4"
        download_section(a.url, s, e, raw)
        write_srt(model, raw, srt)
        render(raw, srt, final)
        raw.unlink()
    print("4/4 done -> out/clip_*.mp4")


if __name__ == "__main__":
    main()
