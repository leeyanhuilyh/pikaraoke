"""Unit tests for ffmpeg module."""

from unittest.mock import MagicMock, patch

import pytest

from pikaraoke.lib.ffmpeg import (
    build_audio_only_ffmpeg_cmd,
    build_video_only_ffmpeg_cmd,
    get_ffmpeg_version,
    get_media_duration,
    is_ffmpeg_installed,
    is_transpose_enabled,
    supports_hardware_h264_encoding,
)


def _make_fr(file_extension=".mp4", cdg_file_path=None, file_path="/songs/track.mp4"):
    mock_fr = MagicMock()
    mock_fr.file_path = file_path
    mock_fr.cdg_file_path = cdg_file_path
    mock_fr.file_extension = file_extension
    return mock_fr


class TestGetFfmpegVersion:
    """Tests for the get_ffmpeg_version function."""

    def test_version_parsed_correctly(self):
        """Test parsing FFmpeg version from output."""
        mock_result = MagicMock()
        mock_result.stdout = "ffmpeg version 5.1.2 Copyright (c) 2000-2022"

        with patch("subprocess.run", return_value=mock_result):
            result = get_ffmpeg_version()
            assert result == "5.1.2"

    def test_ffmpeg_not_installed(self):
        """Test handling when FFmpeg is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = get_ffmpeg_version()
            assert result == "FFmpeg is not installed"

    def test_unable_to_parse_version(self):
        """Test handling when version can't be parsed."""
        mock_result = MagicMock()
        mock_result.stdout = "unexpected format"

        with patch("subprocess.run", return_value=mock_result):
            result = get_ffmpeg_version()
            assert result == "Unable to parse FFmpeg version"


class TestIsTransposeEnabled:
    """Tests for the is_transpose_enabled function."""

    def test_rubberband_available(self):
        """Test when rubberband filter is available."""
        mock_result = MagicMock()
        mock_result.stdout = b"... rubberband ... other filters"

        with patch("subprocess.run", return_value=mock_result):
            assert is_transpose_enabled() is True

    def test_rubberband_not_available(self):
        """Test when rubberband filter is not available."""
        mock_result = MagicMock()
        mock_result.stdout = b"aecho, aresample, volume"

        with patch("subprocess.run", return_value=mock_result):
            assert is_transpose_enabled() is False

    def test_ffmpeg_not_installed(self):
        """Test when FFmpeg is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert is_transpose_enabled() is False


class TestSupportsHardwareH264Encoding:
    """Tests for the supports_hardware_h264_encoding function."""

    def test_x86_returns_false(self):
        """Test that x86 architecture returns False."""
        with patch("platform.machine", return_value="x86_64"):
            assert supports_hardware_h264_encoding() is False

    def test_intel_returns_false(self):
        """Test that Intel architecture returns False."""
        with patch("platform.machine", return_value="i686"):
            assert supports_hardware_h264_encoding() is False

    def test_arm_with_encoder(self):
        """Test ARM with h264_v4l2m2m available."""
        mock_result = MagicMock()
        mock_result.stdout = b"h264_v4l2m2m encoder available"

        with patch("platform.machine", return_value="aarch64"):
            with patch("subprocess.run", return_value=mock_result):
                assert supports_hardware_h264_encoding() is True

    def test_arm_without_encoder(self):
        """Test ARM without h264_v4l2m2m available."""
        mock_result = MagicMock()
        mock_result.stdout = b"libx264 encoder only"

        with patch("platform.machine", return_value="armv7l"):
            with patch("subprocess.run", return_value=mock_result):
                assert supports_hardware_h264_encoding() is False

    def test_arm_ffmpeg_not_found(self):
        """Test ARM when FFmpeg is not installed."""
        with patch("platform.machine", return_value="aarch64"):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                assert supports_hardware_h264_encoding() is False


class TestIsFfmpegInstalled:
    """Tests for the is_ffmpeg_installed function."""

    def test_ffmpeg_installed(self):
        """Test when FFmpeg is installed."""
        with patch("subprocess.run", return_value=MagicMock()):
            assert is_ffmpeg_installed() is True

    def test_ffmpeg_not_installed(self):
        """Test when FFmpeg is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert is_ffmpeg_installed() is False


class TestGetMediaDuration:
    """Tests for the get_media_duration function."""

    def test_returns_duration_rounded(self):
        """Test that duration is returned as rounded integer."""
        with patch("pikaraoke.lib.ffmpeg.ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {"format": {"duration": "183.456"}}
            result = get_media_duration("/path/to/video.mp4")
            assert result == 183

    def test_returns_none_on_probe_error(self):
        """Test that None is returned when probe fails."""
        with patch("pikaraoke.lib.ffmpeg.ffmpeg.probe") as mock_probe:
            mock_probe.side_effect = Exception("Probe failed")
            result = get_media_duration("/path/to/invalid.mp4")
            assert result is None

    def test_returns_none_on_missing_duration(self):
        """Test that None is returned when duration key is missing."""
        with patch("pikaraoke.lib.ffmpeg.ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {"format": {}}
            result = get_media_duration("/path/to/video.mp4")
            assert result is None

    def test_handles_integer_duration(self):
        """Test handling of integer duration value."""
        with patch("pikaraoke.lib.ffmpeg.ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {"format": {"duration": "120"}}
            result = get_media_duration("/path/to/video.mp4")
            assert result == 120


class TestIsTransposeEnabledIndexError:
    """Additional tests for is_transpose_enabled IndexError handling."""

    def test_index_error_returns_false(self):
        """Test that IndexError returns False."""
        with patch("subprocess.run", side_effect=IndexError):
            assert is_transpose_enabled() is False


class TestSupportsHardwareH264EncodingIndexError:
    """Additional tests for supports_hardware_h264_encoding IndexError handling."""

    def test_index_error_returns_false(self):
        """Test that IndexError on ARM returns False."""
        with patch("platform.machine", return_value="aarch64"):
            with patch("subprocess.run", side_effect=IndexError):
                assert supports_hardware_h264_encoding() is False


class TestBuildVideoOnlyFfmpegCmd:
    """Tests for build_video_only_ffmpeg_cmd."""

    def test_no_audio_stream_in_output(self):
        args = build_video_only_ffmpeg_cmd(
            _make_fr(), "/tmp/out_video.m3u8", "/tmp/out_video_%03d.m4s", "out_video_init.mp4"
        ).get_args()

        assert "-acodec" not in args
        assert "rubberband" not in " ".join(args)

    def test_hls_output_args(self):
        args = build_video_only_ffmpeg_cmd(
            _make_fr(), "/tmp/out_video.m3u8", "/tmp/out_video_%03d.m4s", "out_video_init.mp4"
        ).get_args()

        assert "-hls_time" in args
        assert args[args.index("-hls_time") + 1] == "3"
        assert "-force_key_frames" in args
        assert "-hls_fmp4_init_filename" in args
        assert args[args.index("-hls_fmp4_init_filename") + 1] == "out_video_init.mp4"

    def test_mp4_source_copies_video_codec(self):
        args = build_video_only_ffmpeg_cmd(
            _make_fr(file_extension=".mp4"),
            "/tmp/out_video.m3u8",
            "/tmp/out_video_%03d.m4s",
            "out_video_init.mp4",
        ).get_args()

        assert args[args.index("-vcodec") + 1] == "copy"

    def test_raises_without_file_path(self):
        fr = _make_fr()
        fr.file_path = None

        with pytest.raises(ValueError):
            build_video_only_ffmpeg_cmd(fr, "/tmp/out.m3u8", "/tmp/out_%03d.m4s", "out_init.mp4")


class TestBuildAudioOnlyFfmpegCmd:
    """Tests for build_audio_only_ffmpeg_cmd."""

    def test_no_video_stream_in_output(self):
        args = build_audio_only_ffmpeg_cmd(
            _make_fr(),
            4,
            "/tmp/out_audio_p4.m3u8",
            "/tmp/out_audio_p4_%03d.m4s",
            "out_audio_p4_init.mp4",
        ).get_args()

        assert "-vcodec" not in args
        assert "-force_key_frames" not in args

    def test_applies_rubberband_pitch_shift(self):
        args = build_audio_only_ffmpeg_cmd(
            _make_fr(),
            4,
            "/tmp/out_audio_p4.m3u8",
            "/tmp/out_audio_p4_%03d.m4s",
            "out_audio_p4_init.mp4",
        ).get_args()

        filter_arg = args[args.index("-filter_complex") + 1]
        assert "rubberband=pitch=1.259" in filter_arg

    def test_zero_semitones_skips_rubberband(self):
        args = build_audio_only_ffmpeg_cmd(
            _make_fr(),
            0,
            "/tmp/out_audio_p0.m3u8",
            "/tmp/out_audio_p0_%03d.m4s",
            "out_audio_p0_init.mp4",
        ).get_args()

        assert "rubberband" not in " ".join(args)

    def test_playlist_type_is_event_so_partial_renders_are_playable(self):
        # A vod playlist only lands when ffmpeg exits, which would make a
        # half-rendered pitch unswitchable and block playback on a full render.
        args = build_audio_only_ffmpeg_cmd(
            _make_fr(),
            0,
            "/tmp/out_audio_p0.m3u8",
            "/tmp/out_audio_p0_%03d.m4s",
            "out_audio_p0_init.mp4",
        ).get_args()

        assert args[args.index("-hls_playlist_type") + 1] == "event"

    def test_raises_without_file_path(self):
        fr = _make_fr()
        fr.file_path = None

        with pytest.raises(ValueError):
            build_audio_only_ffmpeg_cmd(fr, 0, "/tmp/out.m3u8", "/tmp/out_%03d.m4s", "out_init.mp4")
