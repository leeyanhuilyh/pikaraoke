"""Benchmark vocal removal on this machine.

Measures how fast the app's separator (pikaraoke.lib.mdx) runs here, in time per
second of audio, and its peak memory. By default it takes a minute from the
middle of the song, since intros are often instrumental, and saves that section
both as-is and with the vocals removed, for listening. The model is downloaded
on first use, to the same place the app keeps it.

Usage:
    uv run python scripts/bench_vocal_removal.py SONG [--seconds N] [--threads N] [--out-dir DIR]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

# Add project root to path so we can import pikaraoke modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pikaraoke.lib import mdx  # pylint: disable=wrong-import-position
from pikaraoke.lib.ffmpeg import PCM_FORMAT  # pylint: disable=wrong-import-position
from pikaraoke.lib.get_platform import (  # pylint: disable=wrong-import-position
    get_data_directory,
)


def peak_memory_mb() -> float:
    """Peak memory of this process so far: peak RSS on Linux/macOS, peak working set on Windows."""
    try:
        import resource  # pylint: disable=import-outside-toplevel

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is in KB on Linux but bytes on macOS.
        return peak / 1024 / (1024 if sys.platform == "darwin" else 1)
    except ImportError:
        return psutil.Process().memory_info().peak_wset / 1024 / 1024


def section(path: str, seconds: float) -> tuple[float, float]:
    """Start and length of the section to separate: `seconds` from the middle, or all of it."""
    duration = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    if seconds <= 0 or seconds >= duration:
        return 0.0, duration
    return (duration - seconds) / 2, seconds


def save_wav(path: str, audio: np.ndarray) -> None:
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", PCM_FORMAT, "-ar", str(mdx.SAMPLE_RATE)]
    cmd += ["-ac", str(mdx.CHANNELS), "-i", "-", path]
    subprocess.run(cmd, input=mdx.to_pcm16(audio), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("input", help="song or clip (any format ffmpeg reads)")
    parser.add_argument(
        "--seconds",
        type=float,
        default=60,
        help="length taken from the middle; 0 for the whole song",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out-dir", default=".", help="where the two output files are written")
    args = parser.parse_args()

    model = mdx.ensure_model(os.path.join(get_data_directory(), "models"))
    start, length = section(args.input, args.seconds)
    mix = mdx.decode(args.input, start, length)
    seconds = mix.shape[1] / mdx.SAMPLE_RATE

    started = time.monotonic()
    separator = mdx.MDX(model, args.threads)
    loaded = time.monotonic()
    removed = np.concatenate(list(separator.instrumental(mix)), axis=1)
    finished = time.monotonic()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    original_path = os.path.join(args.out_dir, f"{stem}.original.wav")
    removed_path = os.path.join(args.out_dir, f"{stem}.{mdx.MODEL_NAME}.wav")
    save_wav(original_path, mix)
    save_wav(removed_path, removed)

    work = finished - loaded
    minutes_per_song = (loaded - started + work / seconds * 270) / 60
    print(f"model:            {mdx.MODEL_NAME} ({args.threads} threads)")
    print(f"section:          {seconds:.0f}s starting at {int(start // 60)}:{start % 60:04.1f}")
    print(f"model load:       {loaded - started:.1f}s")
    print(f"separation:       {work:.1f}s  ({work / seconds:.2f}s per second of audio)")
    print(f"4.5-minute song:  ~{minutes_per_song:.1f} minutes (estimated)")
    print(f"peak memory:      {peak_memory_mb():.0f} MB")
    print(f"saved:            {original_path}")
    print(f"                  {removed_path}")


if __name__ == "__main__":
    main()
