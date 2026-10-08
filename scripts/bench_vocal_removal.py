# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "onnxruntime", "psutil"]
# ///
"""Benchmark UVR MDX-Net vocal removal on this machine, without PyTorch.

Measures whether a lighter model than Demucs can separate a song fast enough on
a given machine: time per second of audio, and peak memory. By default it takes
a minute from the middle of the song, since intros are often instrumental, and
saves that section both as-is and with the vocals removed, for listening. The
model and its settings are downloaded on first use. Runs in its own
environment, so the project's dependencies are untouched.

Usage:
    uv run scripts/bench_vocal_removal.py SONG [--seconds N] [--model NAME] [--threads N]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

# Installed from the inline script metadata above, not the project, so pylint can't see it.
import onnxruntime as ort  # pylint: disable=import-error
import psutil

SR = 44100
HOP = 1024
MODELS_URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"
SETTINGS_URL = (
    "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data.json"
)
CACHE = os.path.expanduser("~/.cache/pikaraoke-vocal-bench")


def fetch(url: str, path: str) -> str:
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)}...", file=sys.stderr)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def model_settings(model_path: str) -> dict:
    """UVR's settings table is keyed by an md5 of the model file's last 10MB."""
    with open(model_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 10000 * 1024))
        key = hashlib.md5(f.read()).hexdigest()
    with open(fetch(SETTINGS_URL, os.path.join(CACHE, "model_data.json"))) as f:
        return json.load(f)[key]


class MDX:
    def __init__(self, model_path: str, threads: int) -> None:
        s = model_settings(model_path)
        self.n_fft = s["mdx_n_fft_scale_set"]
        self.dim_f = s["mdx_dim_f_set"]
        self.dim_t = 2 ** s["mdx_dim_t_set"]
        self.compensate = s["compensate"]
        self.model_outputs_vocals = s["primary_stem"] == "Vocals"
        self.chunk = HOP * (self.dim_t - 1)
        self.trim = self.n_fft // 2
        self.step = self.chunk - 2 * self.trim
        n = np.arange(self.n_fft)
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * n / self.n_fft)).astype(np.float32)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        # onnxruntime's memory pool otherwise keeps every chunk's peak working
        # memory reserved, nearly doubling peak RAM for no speed gain.
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _stft(self, x: np.ndarray) -> np.ndarray:
        """(2, chunk) -> (1, 4, dim_f, dim_t), matching torch.stft with center=True."""
        pad = self.n_fft // 2
        x = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
        frames = 1 + (x.shape[1] - self.n_fft) // HOP
        idx = np.arange(self.n_fft)[None, :] + HOP * np.arange(frames)[:, None]
        spec = np.fft.rfft(x[:, idx] * self.window, axis=-1).transpose(0, 2, 1)[:, : self.dim_f]
        return np.stack([spec.real, spec.imag], axis=1).reshape(1, 4, self.dim_f, frames)

    def _istft(self, spec: np.ndarray) -> np.ndarray:
        """Inverse of _stft: (1, 4, dim_f, dim_t) -> (2, chunk)."""
        _, _, f, t = spec.shape
        spec = spec.reshape(2, 2, f, t)
        z = np.zeros((2, self.n_fft // 2 + 1, t), dtype=np.complex64)
        z[:, :f] = spec[:, 0] + 1j * spec[:, 1]
        frames = np.fft.irfft(z.transpose(0, 2, 1), n=self.n_fft, axis=-1) * self.window
        out = np.zeros((2, self.n_fft + HOP * (t - 1)), dtype=np.float32)
        norm = np.zeros(out.shape[1], dtype=np.float32)
        for i in range(t):
            out[:, i * HOP : i * HOP + self.n_fft] += frames[:, i]
            norm[i * HOP : i * HOP + self.n_fft] += self.window**2
        pad = self.n_fft // 2
        keep = slice(pad, pad + HOP * (t - 1))
        return out[:, keep] / np.maximum(norm[keep], 1e-8)

    def instrumental(self, mix: np.ndarray):
        """Yield the vocals-removed audio of `mix` (2, N) one chunk at a time."""
        n = mix.shape[1]
        tail = self.step - n % self.step
        padded = np.pad(mix, ((0, 0), (self.trim, tail + self.trim)))
        for start in range(0, n, self.step):
            wave = padded[:, start : start + self.chunk]
            pred = self.session.run(None, {self.input_name: self._stft(wave)})[0]
            part = self._istft(pred)[:, self.trim : -self.trim][:, : n - start] * self.compensate
            yield (
                part
                if not self.model_outputs_vocals
                else mix[:, start : start + part.shape[1]] - part
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


def decode(path: str, start: float, length: float) -> np.ndarray:
    """Decode a section of any audio or video file to stereo float32 at 44.1kHz, shaped (2, N)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", path]
    cmd += ["-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()


def encoder(path: str) -> subprocess.Popen:
    """An ffmpeg process that writes raw stereo float32 from its stdin to `path`."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-"]
    return subprocess.Popen(cmd + [path], stdin=subprocess.PIPE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("input", help="song or clip (any format ffmpeg reads)")
    parser.add_argument(
        "--seconds",
        type=float,
        default=60,
        help="length taken from the middle; 0 for the whole song",
    )
    parser.add_argument("--model", default="UVR_MDXNET_9482", help="UVR MDX-Net model name")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out-dir", default=".", help="where the two output files are written")
    args = parser.parse_args()

    model = fetch(f"{MODELS_URL}/{args.model}.onnx", os.path.join(CACHE, f"{args.model}.onnx"))
    start, length = section(args.input, args.seconds)
    mix = decode(args.input, start, length)
    seconds = mix.shape[1] / SR

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    original_path = os.path.join(args.out_dir, f"{stem}.original.wav")
    removed_path = os.path.join(args.out_dir, f"{stem}.{args.model}.wav")
    original = encoder(original_path)
    original.communicate(np.ascontiguousarray(mix.T).tobytes())

    started = time.monotonic()
    separator = MDX(model, args.threads)
    loaded = time.monotonic()
    removed = encoder(removed_path)
    for part in separator.instrumental(mix):
        removed.stdin.write(np.ascontiguousarray(part.T, dtype=np.float32).tobytes())
    removed.stdin.close()
    removed.wait()
    finished = time.monotonic()

    work = finished - loaded
    minutes_per_song = (loaded - started + work / seconds * 270) / 60
    print(f"model:            {args.model} ({args.threads} threads)")
    print(f"section:          {seconds:.0f}s starting at {int(start // 60)}:{start % 60:04.1f}")
    print(f"model load:       {loaded - started:.1f}s")
    print(f"separation:       {work:.1f}s  ({work / seconds:.2f}s per second of audio)")
    print(f"4.5-minute song:  ~{minutes_per_song:.1f} minutes (estimated)")
    print(f"peak memory:      {peak_memory_mb():.0f} MB")
    print(f"saved:            {original_path}")
    print(f"                  {removed_path}")


if __name__ == "__main__":
    main()
