"""FFmpeg utilities for media processing and transcoding."""

from __future__ import annotations

import logging
import platform
import subprocess
from typing import TYPE_CHECKING, Any

import ffmpeg

from pikaraoke.lib.get_platform import is_running_in_docker

if TYPE_CHECKING:
    from pikaraoke.lib.file_resolver import FileResolver

HLS_SEGMENT_SECONDS = 3


def _ffmpeg_input(file_path: str, file_extension: str):
    """Build an ffmpeg input, adding genpts for containers with VFR/timestamp issues."""
    if file_extension in [".webm", ".avi", ".mov", ".mkv"]:
        return ffmpeg.input(file_path, **{"fflags": "+genpts"})
    return ffmpeg.input(file_path)


def _video_codec_and_bitrate(is_cdg: bool, file_extension: str) -> tuple[str, str]:
    """Pick video codec and bitrate for a transcode.

    CDG always needs encoding; MP4 can copy video stream (already H.264
    compatible). WEBM uses VP8/VP9 which must be transcoded to H.264 for
    fMP4 containers. Pi 3B+ struggles with 5M in real-time, 2M provides
    better stability with the hardware encoder.
    """
    using_hardware_encoder = supports_hardware_h264_encoding()
    default_vcodec = "h264_v4l2m2m" if using_hardware_encoder else "libx264"

    if is_cdg:
        vcodec = "libx264"
    else:
        vcodec = "copy" if file_extension == ".mp4" else default_vcodec

    if is_cdg:
        vbitrate = "500k"
    elif using_hardware_encoder:
        vbitrate = "2M"
    else:
        vbitrate = "5M"

    return vcodec, vbitrate


def _apply_audio_filters(audio, semitones: int, avsync: float, normalize_audio: bool):
    """Apply avsync, pitch-shift, and loudness-normalization filters, in that order."""
    if avsync > 0:
        audio = audio.filter("adelay", f"{avsync * 1000}|{avsync * 1000}")
    elif avsync < 0:
        audio = audio.filter("atrim", start=-avsync)

    if semitones != 0:
        # pitchq=speed is already librubberband's default on recent ffmpeg
        # builds, but older builds (e.g. what a Raspberry Pi OS repo ships)
        # may default elsewhere - set it explicitly rather than assume.
        audio = audio.filter("rubberband", pitch=2 ** (semitones / 12), pitchq="speed")

    if normalize_audio:
        audio = audio.filter("loudnorm", i=-16, tp=-1.5, lra=11)

    return audio


def get_media_duration(file_path: str) -> int | None:
    """Get the duration of a media file in seconds.

    Args:
        file_path: Path to the media file.

    Returns:
        Duration in seconds (rounded), or None if unable to determine.
    """
    try:
        duration = ffmpeg.probe(file_path)["format"]["duration"]
        return round(float(duration))
    except:
        return None


def build_ffmpeg_cmd(
    fr: FileResolver,
    semitones: int = 0,
    normalize_audio: bool = True,
    force_mp4_encoding: bool = False,
    buffer_fully_before_playback: bool = False,
    avsync: float = 0,
    cdg_pixel_scaling: bool = False,
) -> Any:
    """Build an ffmpeg command for transcoding media.

    Handles video/audio codec selection, pitch shifting, audio normalization,
    and CDG file rendering.

    Args:
        fr: FileResolver instance with source file information.
        semitones: Number of semitones to shift pitch (0 = no shift).
        normalize_audio: Whether to apply loudness normalization.
        force_mp4_encoding: If True, force mp4 encoding.
        avsync: Audio/video sync adjustment in seconds.
        cdg_pixel_scaling: Enable pixel scaling for CDG rendering.

    Returns:
        ffmpeg stream object ready to execute with run_async().
    """
    avsync = float(avsync)
    is_cdg = fr.cdg_file_path is not None
    is_transposed = semitones != 0

    if fr.file_path is None:
        raise ValueError("File path is required to build ffmpeg command")

    vcodec, vbitrate = _video_codec_and_bitrate(is_cdg, fr.file_extension or "")

    # Copy audio if no processing needed, otherwise re-encode with AAC
    # CDG always re-encodes audio for compatibility
    acodec = "aac" if is_cdg or is_transposed or normalize_audio or avsync != 0 else "copy"

    input = _ffmpeg_input(fr.file_path, fr.file_extension or "")
    audio = _apply_audio_filters(input.audio, semitones, avsync, normalize_audio)

    # Video source: CDG input or original video stream
    if is_cdg:
        logging.info("Playing CDG/MP3 file: " + fr.file_path)
        cdg_input = ffmpeg.input(fr.cdg_file_path, copyts=None)
        video = cdg_input.video.filter("fps", fps=25)
        if cdg_pixel_scaling:
            video = video.filter("scale", -1, 720, flags="neighbor")
    else:
        video = input.video

    # Build output based on format
    if force_mp4_encoding:
        movflags = (
            "+faststart" if buffer_fully_before_playback else "frag_keyframe+default_base_moof"
        )
        output = ffmpeg.output(
            audio,
            video,
            fr.output_file,
            vcodec=vcodec,
            acodec=acodec,
            preset="ultrafast",
            listen=1,
            f="mp4",
            video_bitrate=vbitrate,
            movflags=movflags,
            **({"pix_fmt": "yuv420p"} if is_cdg else {}),
        )
    else:
        # HLS format with fMP4 segments
        # Both MP4 and HLS streaming modes use this - difference is in serving:
        # - mp4: Stream concatenates init + segments for progressive playback
        # - hls: Browser requests segments via .m3u8 playlist
        output = ffmpeg.output(
            audio,
            video,
            fr.output_file,
            vcodec=vcodec,
            acodec="aac",
            audio_bitrate="192k",
            ac=2,  # Force stereo
            ar=48000,  # Standard sample rate
            preset="ultrafast",
            f="hls",
            hls_time=HLS_SEGMENT_SECONDS,
            hls_list_size=0,
            hls_playlist_type="event",
            hls_segment_type="fmp4",
            hls_fmp4_init_filename=fr.init_filename,
            hls_segment_filename=fr.segment_pattern,
            video_bitrate=vbitrate,
            # Without an explicit keyframe interval, segments can only be cut
            # at whatever keyframe interval the encoder defaults to (often
            # 8-10s), so hls_time alone is a no-op and segments end up 2-3x
            # longer than intended. No effect when video is stream-copied.
            force_key_frames=f"expr:gte(t,n_forced*{HLS_SEGMENT_SECONDS})",
            # CDG needs pix_fmt for proper color space
            **({"pix_fmt": "yuv420p"} if is_cdg else {}),
            **{
                "fps_mode": "cfr",
                "avoid_negative_ts": "make_zero",
            },
        )

    args = output.get_args()
    logging.debug(f"COMMAND: ffmpeg " + " ".join(args))
    return output


def build_video_only_ffmpeg_cmd(
    fr: FileResolver,
    output_file: str,
    segment_filename: str,
    init_filename: str,
    cdg_pixel_scaling: bool = False,
) -> Any:
    """Build an ffmpeg command for a video-only HLS rendition.

    Used by pitch pre-rendering: video never changes with pitch, so it's
    transcoded once and shared across every pitch rendition instead of
    being re-encoded per semitone.

    Args:
        fr: FileResolver instance with source file information.
        output_file: Path to write the HLS playlist (.m3u8) to.
        segment_filename: hls_segment_filename pattern for this rendition.
        init_filename: hls_fmp4_init_filename for this rendition.
        cdg_pixel_scaling: Enable pixel scaling for CDG rendering.

    Returns:
        ffmpeg stream object ready to execute with run_async().
    """
    is_cdg = fr.cdg_file_path is not None

    if fr.file_path is None:
        raise ValueError("File path is required to build ffmpeg command")

    vcodec, vbitrate = _video_codec_and_bitrate(is_cdg, fr.file_extension or "")

    if is_cdg:
        logging.info("Playing CDG/MP3 file: " + fr.file_path)
        cdg_input = ffmpeg.input(fr.cdg_file_path, copyts=None)
        video = cdg_input.video.filter("fps", fps=25)
        if cdg_pixel_scaling:
            video = video.filter("scale", -1, 720, flags="neighbor")
    else:
        video = _ffmpeg_input(fr.file_path, fr.file_extension or "").video

    output = ffmpeg.output(
        video,
        output_file,
        vcodec=vcodec,
        preset="ultrafast",
        f="hls",
        hls_time=HLS_SEGMENT_SECONDS,
        hls_list_size=0,
        hls_playlist_type="event",
        hls_segment_type="fmp4",
        hls_fmp4_init_filename=init_filename,
        hls_segment_filename=segment_filename,
        video_bitrate=vbitrate,
        force_key_frames=f"expr:gte(t,n_forced*{HLS_SEGMENT_SECONDS})",
        **({"pix_fmt": "yuv420p"} if is_cdg else {}),
        **{"fps_mode": "cfr", "avoid_negative_ts": "make_zero"},
    )

    args = output.get_args()
    logging.debug(f"COMMAND: ffmpeg " + " ".join(args))
    return output


def build_audio_only_ffmpeg_cmd(
    fr: FileResolver,
    semitones: int,
    output_file: str,
    segment_filename: str,
    init_filename: str,
    normalize_audio: bool = True,
    avsync: float = 0,
) -> Any:
    """Build an ffmpeg command for an audio-only HLS rendition at a given pitch.

    Used by pitch pre-rendering to render one alternate-audio HLS rendition
    per cached semitone. Runs to completion rather than staying open for
    live control, so no -re pacing is needed - it only has to keep pace
    with a background render, not real-time playback.

    The playlist is written incrementally (event, not vod) so a rendition
    can be switched to while it is still rendering: a vod playlist only
    lands when ffmpeg exits, which would make a half-rendered pitch
    unplayable and force playback to wait for a full render.

    Args:
        fr: FileResolver instance with source file information.
        semitones: Number of semitones to shift pitch (0 = no shift).
        output_file: Path to write the HLS playlist (.m3u8) to.
        segment_filename: hls_segment_filename pattern for this rendition.
        init_filename: hls_fmp4_init_filename for this rendition.
        normalize_audio: Whether to apply loudness normalization.
        avsync: Audio/video sync adjustment in seconds.

    Returns:
        ffmpeg stream object ready to execute with run_async().
    """
    avsync = float(avsync)

    if fr.file_path is None:
        raise ValueError("File path is required to build ffmpeg command")

    audio = _ffmpeg_input(fr.file_path, fr.file_extension or "").audio
    audio = _apply_audio_filters(audio, semitones, avsync, normalize_audio)

    output = ffmpeg.output(
        audio,
        output_file,
        acodec="aac",
        audio_bitrate="192k",
        ac=2,
        ar=48000,
        f="hls",
        hls_time=HLS_SEGMENT_SECONDS,
        hls_list_size=0,
        hls_playlist_type="event",
        hls_segment_type="fmp4",
        hls_fmp4_init_filename=init_filename,
        hls_segment_filename=segment_filename,
    )

    args = output.get_args()
    logging.debug(f"COMMAND: ffmpeg " + " ".join(args))
    return output


def get_ffmpeg_version() -> str:
    """Get the installed FFmpeg version string.

    Returns:
        Version string, or an error message if FFmpeg is not installed
        or version cannot be parsed.
    """
    try:
        # Execute the command 'ffmpeg -version'
        result = subprocess.run(
            ["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        # Parse the first line to get the version
        first_line = result.stdout.split("\n")[0]
        version_info = first_line.split(" ")[2]  # Assumes the version info is the third element
        return version_info
    except FileNotFoundError:
        return "FFmpeg is not installed"
    except IndexError:
        return "Unable to parse FFmpeg version"


def is_transpose_enabled() -> bool:
    """Check if FFmpeg has the rubberband filter for pitch shifting.

    Returns:
        True if rubberband filter is available, False otherwise.
    """
    try:
        filters = subprocess.run(["ffmpeg", "-filters"], capture_output=True)
    except FileNotFoundError:
        return False
    except IndexError:
        return False
    return "rubberband" in filters.stdout.decode()


def supports_hardware_h264_encoding() -> bool:
    """Check if hardware H.264 encoding (h264_v4l2m2m) is available.

    Only returns True on ARM architecture (Raspberry Pi) where h264_v4l2m2m
    is actually supported. On x86/Intel systems, returns False to use software encoding.

    Returns:
        True if hardware encoding is available, False otherwise.
    """
    # Check CPU architecture first - h264_v4l2m2m only works on ARM
    arch = platform.machine().lower()
    is_arm = any(arm_variant in arch for arm_variant in ["arm", "aarch"])

    if not is_arm:
        # Not ARM (probably Intel x86/x64), don't use h264_v4l2m2m
        logging.debug(f"CPU architecture {arch} is not ARM, using software encoder")
        return False

    if is_running_in_docker():
        # Docker containers do not have access to the GPU
        logging.debug("Running in Docker where GPU access is not available, using software encoder")
        return False

    # On ARM, check if h264_v4l2m2m is available
    try:
        codecs = subprocess.run(["ffmpeg", "-codecs"], capture_output=True)
    except FileNotFoundError:
        return False
    except IndexError:
        return False

    has_encoder = "h264_v4l2m2m" in codecs.stdout.decode()
    if has_encoder:
        logging.info("ARM platform detected, using h264_v4l2m2m hardware encoder")
    else:
        logging.debug("ARM platform but h264_v4l2m2m not available")

    return has_encoder


def is_ffmpeg_installed() -> bool:
    """Check if FFmpeg is installed and accessible.

    Returns:
        True if FFmpeg is installed, False otherwise.
    """
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True)
    except FileNotFoundError:
        return False
    return True
