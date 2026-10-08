"""Unit tests for the MDX vocal separator, with the model itself stubbed out."""

import hashlib
import io
from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")

from pikaraoke.lib import mdx  # pylint: disable=wrong-import-position


@pytest.fixture
def separator():
    with patch("pikaraoke.lib.mdx.ort.InferenceSession"):
        return mdx.MDX("model.onnx", threads=1)


def song(seconds: float) -> np.ndarray:
    """Stereo noise, a length that is not a whole number of model chunks."""
    rng = np.random.default_rng(0)
    return (rng.standard_normal((2, int(seconds * mdx.SAMPLE_RATE))) * 0.1).astype(np.float32)


class TestInstrumental:
    def test_chunks_cover_the_song_exactly_once_and_in_order(self, separator):
        """With no vocals found, every chunk must hand back its slice of the mix untouched:
        any gap, overlap or shift between chunks would show up as a difference."""
        mix = song(17.3)
        silent = lambda spec: np.zeros_like(spec)  # noqa: E731

        with patch.object(separator, "_predict", side_effect=silent):
            out = np.concatenate(list(separator.instrumental(mix)), axis=1)

        assert out.shape == mix.shape
        np.testing.assert_allclose(out, mix, atol=1e-6)

    def test_spectrogram_round_trips_back_to_the_same_audio(self, separator):
        """If the model called everything vocals, the vocals it hands back must be the
        mix itself. Anything else means the spectrogram and its inverse don't match up.

        Tones below the model's 14.7kHz cutoff, since it never sees anything above.
        """
        t = np.arange(int(13.0 * mdx.SAMPLE_RATE)) / mdx.SAMPLE_RATE
        tones = sum(np.sin(2 * np.pi * f * t) / 5 for f in (110, 440, 1250, 3700, 9100))
        mix = np.stack([tones, np.roll(tones, 1000)]).astype(np.float32)

        with patch.object(separator, "_predict", side_effect=lambda spec: spec):
            out = np.concatenate(list(separator.instrumental(mix)), axis=1)

        vocals = (mix - out) / mdx.COMPENSATE
        error = np.sqrt(np.mean((vocals - mix) ** 2) / np.mean(mix**2))
        assert error < 1e-3  # -60dB: inaudible

    def test_yields_chunks_in_playback_order(self, separator):
        """A reader follows the output as it is written, so chunks must arrive in order."""
        mix = np.tile(np.arange(int(12 * mdx.SAMPLE_RATE), dtype=np.float32), (2, 1))
        with patch.object(separator, "_predict", side_effect=lambda spec: np.zeros_like(spec)):
            firsts = [chunk[0, 0] for chunk in separator.instrumental(mix)]
        assert firsts == sorted(firsts)


class TestToPcm16:
    def test_interleaves_left_and_right(self):
        part = np.array([[0.5, -0.5], [0.25, -0.25]], dtype=np.float32)
        samples = np.frombuffer(mdx.to_pcm16(part), dtype="<i2")
        assert list(samples) == [16383, 8191, -16383, -8191]

    def test_clips_rather_than_wrapping_around(self):
        """Removing vocals can push peaks past full scale; wrapping would be a loud click."""
        part = np.array([[1.5], [-1.5]], dtype=np.float32)
        assert list(np.frombuffer(mdx.to_pcm16(part), dtype="<i2")) == [32767, -32767]


class TestEnsureModel:
    def test_reuses_a_model_already_downloaded(self, tmp_path):
        (tmp_path / f"{mdx.MODEL_NAME}.onnx").write_bytes(b"model")
        with patch("pikaraoke.lib.mdx.urllib.request.urlopen") as urlopen:
            assert mdx.ensure_model(str(tmp_path)).endswith(".onnx")
        urlopen.assert_not_called()

    def test_keeps_a_download_that_matches_the_checksum(self, tmp_path):
        data = b"the real model"
        with patch("pikaraoke.lib.mdx.MODEL_SHA256", hashlib.sha256(data).hexdigest()):
            with patch("pikaraoke.lib.mdx.urllib.request.urlopen", return_value=io.BytesIO(data)):
                path = mdx.ensure_model(str(tmp_path))
        with open(path, "rb") as f:
            assert f.read() == data

    def test_rejects_a_download_that_does_not_match(self, tmp_path):
        """A truncated or tampered model must not be cached as if it were the real one."""
        with patch("pikaraoke.lib.mdx.urllib.request.urlopen", return_value=io.BytesIO(b"bad")):
            with pytest.raises(RuntimeError):
                mdx.ensure_model(str(tmp_path))
        assert list(tmp_path.iterdir()) == []
