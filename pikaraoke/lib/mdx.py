"""Vocal removal with the UVR MDX-Net model UVR_MDXNET_9482, via onnxruntime.

Runs as its own process (``python -m pikaraoke.lib.mdx``) so separation stays
off the web server's event loop and can be given a lower CPU priority. Needs
numpy and onnxruntime from the optional vocal-separation dependency group,
which is why the app never imports this module, only runs it.

The output is raw 16-bit stereo PCM at 44.1kHz, appended one model chunk at a
time, so a reader can follow it while it is still being written.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 44100
CHANNELS = 2
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * 2  # 16-bit samples

MODEL_NAME = "UVR_MDXNET_9482"
MODEL_URL = (
    "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
    f"{MODEL_NAME}.onnx"
)
MODEL_SHA256 = "f4f365207c56deb115bceedff3ad8fe98a751c745f9e370cecec6226b8b47184"

# This model's settings, from UVR's model_data.json (keyed by an md5 of the
# file's last 10MB). The network was trained on spectrograms made exactly
# this way, so none of these can be changed independently.
N_FFT = 6144
HOP = 1024
DIM_F = 2048
DIM_T = 256
COMPENSATE = 1.035  # the model's output runs slightly quiet


def ensure_model(model_dir: str) -> str:
    """Path to the model, downloading and checksumming it on first use."""
    path = os.path.join(model_dir, f"{MODEL_NAME}.onnx")
    if os.path.exists(path):
        return path
    os.makedirs(model_dir, exist_ok=True)
    partial = path + ".part"
    print(f"Downloading {MODEL_NAME} model", file=sys.stderr)
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response, open(partial, "wb") as f:
        digest = hashlib.sha256()
        while block := response.read(1 << 20):
            digest.update(block)
            f.write(block)
    if digest.hexdigest() != MODEL_SHA256:
        os.remove(partial)
        raise RuntimeError(f"Downloaded {MODEL_NAME} does not match its expected checksum")
    os.replace(partial, path)
    return path


class MDX:
    """One loaded MDX-Net model, separating audio a chunk at a time."""

    def __init__(self, model_path: str, threads: int) -> None:
        self.chunk = HOP * (DIM_T - 1)
        # STFT edges are unreliable, so each chunk overlaps its neighbours by
        # this much on both sides and only its middle is kept.
        self.trim = N_FFT // 2
        self.step = self.chunk - 2 * self.trim
        n = np.arange(N_FFT)
        # Periodic Hann, as torch.hann_window used in training.
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)).astype(np.float32)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        # The default memory pool keeps every chunk's peak working memory
        # reserved, nearly doubling peak RAM for no speed gain.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self.session = ort.InferenceSession(
            model_path, options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def _spectrogram(self, wave: np.ndarray) -> np.ndarray:
        """(2, chunk) -> (1, 4, DIM_F, DIM_T): left and right, each as real and imaginary planes.

        Matches torch.stft with center=True, which the model was trained on.
        """
        pad = N_FFT // 2
        wave = np.pad(wave, ((0, 0), (pad, pad)), mode="reflect")
        frames = 1 + (wave.shape[1] - N_FFT) // HOP
        index = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
        spec = np.fft.rfft(wave[:, index] * self.window, axis=-1).transpose(0, 2, 1)[:, :DIM_F]
        return np.stack([spec.real, spec.imag], axis=1).reshape(1, 4, DIM_F, frames)

    def _wave(self, spec: np.ndarray) -> np.ndarray:
        """Inverse of _spectrogram: (1, 4, DIM_F, DIM_T) -> (2, chunk)."""
        frames = spec.shape[-1]
        spec = spec.reshape(2, 2, DIM_F, frames)
        full = np.zeros((2, N_FFT // 2 + 1, frames), dtype=np.complex64)
        full[:, :DIM_F] = spec[:, 0] + 1j * spec[:, 1]
        pieces = np.fft.irfft(full.transpose(0, 2, 1), n=N_FFT, axis=-1) * self.window
        out = np.zeros((2, N_FFT + HOP * (frames - 1)), dtype=np.float32)
        norm = np.zeros(out.shape[1], dtype=np.float32)
        for i in range(frames):
            out[:, i * HOP : i * HOP + N_FFT] += pieces[:, i]
            norm[i * HOP : i * HOP + N_FFT] += self.window**2
        keep = slice(N_FFT // 2, N_FFT // 2 + HOP * (frames - 1))
        return out[:, keep] / np.maximum(norm[keep], 1e-8)

    def _predict(self, spec: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: spec})[0]

    def instrumental(self, mix: np.ndarray):
        """Yield the vocals-removed audio of `mix` (2, N), in order, one chunk at a time.

        The model estimates the vocals, which are then taken away from the mix.
        """
        n = mix.shape[1]
        tail = self.step - n % self.step
        padded = np.pad(mix, ((0, 0), (self.trim, tail + self.trim)))
        for start in range(0, n, self.step):
            wave = padded[:, start : start + self.chunk]
            vocals = self._wave(self._predict(self._spectrogram(wave)))
            vocals = vocals[:, self.trim : -self.trim][:, : n - start] * COMPENSATE
            yield mix[:, start : start + vocals.shape[1]] - vocals


def decode(path: str, start: float = 0, length: float | None = None) -> np.ndarray:
    """Any audio or video file's audio as stereo float32 at 44.1kHz, shaped (2, N).

    `start` and `length` (seconds) take just a section, e.g. for benchmarking.
    """
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}"]
    if length is not None:
        cmd += ["-t", f"{length:.3f}"]
    cmd += ["-i", path, "-vn", "-ac", str(CHANNELS)]
    cmd += ["-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, CHANNELS).T.copy()


def to_pcm16(part: np.ndarray) -> bytes:
    """(2, N) float audio to interleaved 16-bit PCM bytes, clipping anything out of range."""
    return (np.clip(part.T, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove the vocals from a song.")
    parser.add_argument("input", help="song file (any format ffmpeg reads)")
    parser.add_argument("output", help="raw 16-bit stereo PCM at 44.1kHz is written here")
    parser.add_argument("--model-dir", required=True, help="where the model is cached")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()

    separator = MDX(ensure_model(args.model_dir), args.threads)
    mix = decode(args.input)
    with open(args.output, "wb") as out:
        for part in separator.instrumental(mix):
            out.write(to_pcm16(part))
            out.flush()


if __name__ == "__main__":
    main()
