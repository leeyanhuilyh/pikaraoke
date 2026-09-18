"""FFmpeg utilities for media processing and transcoding."""

from __future__ import annotations

import logging
import platform
import subprocess
from typing import TYPE_CHECKING

import ffmpeg

from pikaraoke.lib.get_platform import is_running_in_docker

if TYPE_CHECKING:
    from pikaraoke.lib.file_resolver import FileResolver

HLS_SEGMENT_SECONDS = 3


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
    pitch_control_port: int,
    normalize_audio: bool = True,
    force_mp4_encoding: bool = False,
    buffer_fully_before_playback: bool = False,
    avsync: float = 0,
    cdg_pixel_scaling: bool = False,
) -> list[str]:
    """Build an ffmpeg command for transcoding media.

    Handles video/audio codec selection, pitch shifting, audio normalization,
    and CDG file rendering. Rubberband pitch-shifting and an azmq control
    socket are always in the audio graph (starting at pitch=1.0, i.e. no
    shift) so the singer's pitch can be changed live via zmq commands while
    the stream keeps playing, instead of restarting ffmpeg.

    Args:
        fr: FileResolver instance with source file information.
        pitch_control_port: Localhost TCP port the azmq filter binds to, for
            sending live 'rubberband pitch <value>' commands.
        normalize_audio: Whether to apply loudness normalization.
        force_mp4_encoding: If True, force mp4 encoding.
        avsync: Audio/video sync adjustment in seconds.
        cdg_pixel_scaling: Enable pixel scaling for CDG rendering.

    Returns:
        Full ffmpeg command-line argument list (excluding the 'ffmpeg' executable).
    """
    avsync = float(avsync)
    is_cdg = fr.cdg_file_path is not None

    if fr.file_path is None:
        raise ValueError("File path is required to build ffmpeg command")

    # Use h/w acceleration on Pi
    using_hardware_encoder = supports_hardware_h264_encoding()
    default_vcodec = "h264_v4l2m2m" if using_hardware_encoder else "libx264"

    # CDG always needs encoding; MP4 can copy video stream (already H.264 compatible)
    # WEBM uses VP8/VP9 which must be transcoded to H.264 for fMP4 containers
    if is_cdg:
        vcodec = "libx264"
    else:
        vcodec = "copy" if fr.file_extension == ".mp4" else default_vcodec

    # Optimize bitrate: CDG is simple graphics (500k), video files need more
    # Pi 3B+ struggles with 5M in real-time, 2M provides better stability
    if is_cdg:
        vbitrate = "500k"
    elif using_hardware_encoder:
        vbitrate = "2M"
    else:
        vbitrate = "5M"

    # Audio always runs through rubberband (for live pitch control), so it
    # always needs re-encoding - never eligible for stream copy.
    acodec = "aac"

    # Read the input at its native frame rate instead of as fast as
    # possible. Without this, ffmpeg races through the whole song's audio
    # in the first few seconds (audio processing is cheap next to video
    # encoding), and libavfilter stops scheduling the azmq node once it has
    # no more audio to pull through - the pitch-control socket goes dead
    # for the rest of the song even though ffmpeg itself is still running.
    # For container formats with VFR or timestamp issues, also use genpts.
    if fr.file_extension in [".webm", ".avi", ".mov", ".mkv"]:
        input = ffmpeg.input(fr.file_path, re=None, fflags="+genpts")
    else:
        input = ffmpeg.input(fr.file_path, re=None)
    audio = input.audio

    # Audio sync adjustment: delay or trim
    if avsync > 0:
        audio = audio.filter("adelay", f"{avsync * 1000}|{avsync * 1000}")
    elif avsync < 0:
        audio = audio.filter("atrim", start=-avsync)

    # Always present so pitch can be changed live later via the azmq
    # command socket, without restarting ffmpeg. Starts untransposed.
    audio = audio.filter("rubberband", pitch=1.0)

    # Loudness normalization
    if normalize_audio:
        audio = audio.filter("loudnorm", i=-16, tp=-1.5, lra=11)

    # Video source: CDG input or original video stream
    if is_cdg:
        logging.info("Playing CDG/MP3 file: " + fr.file_path)
        cdg_input = ffmpeg.input(fr.cdg_file_path, re=None, copyts=None)
        video = cdg_input.video.filter("fps", fps=25)
        if cdg_pixel_scaling:
            video = video.filter("scale", -1, 720, flags="neighbor")
    else:
        video = input.video

    # Force a keyframe every HLS_SEGMENT_SECONDS: without this, segments can
    # only be cut at whatever keyframe interval the encoder defaults to
    # (often 8-10s), so hls_time is a no-op and segments end up 2-3x longer
    # than intended - directly inflating both startup buffering time and how
    # long a live pitch change takes to reach the player. No effect when the
    # video stream is copied rather than re-encoded (e.g. MP4 sources).
    force_key_frames = f"expr:gte(t,n_forced*{HLS_SEGMENT_SECONDS})"

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
            force_key_frames=force_key_frames,
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
            force_key_frames=force_key_frames,
            f="hls",
            hls_time=HLS_SEGMENT_SECONDS,
            hls_list_size=0,
            hls_playlist_type="event",
            hls_segment_type="fmp4",
            hls_fmp4_init_filename=fr.init_filename,
            hls_segment_filename=fr.segment_pattern,
            video_bitrate=vbitrate,
            # CDG needs pix_fmt for proper color space
            **({"pix_fmt": "yuv420p"} if is_cdg else {}),
            **{
                "fps_mode": "cfr",
                "avoid_negative_ts": "make_zero",
            },
        )

    args = _inject_pitch_control_socket(output.get_args(), pitch_control_port)
    logging.debug(f"COMMAND: ffmpeg " + " ".join(args))
    return args


def _inject_pitch_control_socket(args: list[str], port: int) -> list[str]:
    """Splice an azmq command socket in front of the rubberband filter.

    ffmpeg-python has no way to emit a raw, correctly quoted filter option:
    it backslash-escapes every ':' in a value but never wraps the result in
    the single quotes ffmpeg's filtergraph parser needs to treat
    "tcp\\://host\\:port" as one value instead of splitting it on each
    colon, so bind_address has to be spliced in after the fact. rubberband's
    pitch is one of the few audio filter options ffmpeg actually wires up to
    accept commands at runtime, which is what makes live pitch changes
    possible without restarting ffmpeg.
    """
    filter_index = args.index("-filter_complex") + 1
    zmq_filter = f"azmq=bind_address='tcp\\://127.0.0.1\\:{port}'"
    args[filter_index] = args[filter_index].replace(
        "rubberband=pitch=1.0", f"{zmq_filter}[pitchctl];[pitchctl]rubberband=pitch=1.0"
    )
    return args


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
    """Check if FFmpeg supports live pitch shifting.

    Requires both the rubberband filter (pitch shifting) and the azmq
    filter (the command socket used to change pitch live, without
    restarting ffmpeg). azmq requires FFmpeg to be built with
    --enable-libzmq, which is opt-in and often missing from distro
    packages (e.g. Raspberry Pi OS's apt ffmpeg).

    Returns:
        True if both filters are available, False otherwise.
    """
    try:
        filters = subprocess.run(["ffmpeg", "-filters"], capture_output=True)
    except FileNotFoundError:
        return False
    except IndexError:
        return False
    output = filters.stdout.decode()
    return "rubberband" in output and "azmq" in output


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
