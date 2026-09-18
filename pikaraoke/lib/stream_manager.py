"""Stream manager for handling video transcoding and playback setup."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any

import zmq

from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.ffmpeg import build_ffmpeg_cmd
from pikaraoke.lib.file_resolver import FileResolver
from pikaraoke.lib.preference_manager import PreferenceManager
from pikaraoke.lib.url_prefix import normalize_url_base_path


def _find_free_port() -> int:
    """Find an available localhost TCP port for the pitch-control zmq socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class PlaybackResult:
    """Result of a playback operation.

    Attributes:
        success: Whether playback started successfully.
        stream_url: URL path for the video stream.
        subtitle_url: URL path for subtitles (if present).
        duration: Video duration in seconds.
        error: Error message if playback failed.
    """

    success: bool
    stream_url: str | None = None
    subtitle_url: str | None = None
    duration: int | None = None
    error: str | None = None


def enqueue_output(out: Any, queue: Queue) -> None:
    """Read lines from a stream and put them in a queue without blocking.

    Args:
        out: File-like object to read from (e.g., subprocess stderr).
        queue: Queue to put the read lines into.
    """
    for line in iter(out.readline, b""):
        queue.put(line)
    out.close()


class StreamManager:
    """Manages video transcoding and stream preparation for playback.

    Handles FFmpeg transcoding, buffering monitoring, and stream URL setup
    for both HLS and progressive MP4 streaming formats.

    Attributes:
        preferences: PreferenceManager for configuration.
        ffmpeg_process: Currently running FFmpeg subprocess.
        ffmpeg_log: Queue for FFmpeg stderr output.
        pitch_control_port: Localhost port the running process's azmq
            command socket is bound to, or None if nothing is playing.
    """

    def __init__(
        self,
        preferences: PreferenceManager,
        streaming_format: str = "hls",
        base_path: str = "",
    ) -> None:
        """Initialize the stream manager.

        Args:
            preferences: PreferenceManager instance for configuration.
            streaming_format: Video streaming format ('hls' or 'mp4').
            base_path: URL path prefix when PiKaraoke is hosted under a subpath.
        """
        self.preferences = preferences
        self.streaming_format = streaming_format
        self.ffmpeg_process = None
        self.ffmpeg_log: Queue | None = None
        self.base_path = normalize_url_base_path(base_path)
        self.pitch_control_port: int | None = None
        self._zmq_context = zmq.Context()
        # Which process's unexpected exit has already been logged, so a crash
        # after playback starts (nothing else polls the process by then) is
        # reported exactly once instead of silently going unnoticed.
        self._exit_logged_for: subprocess.Popen | None = None

    def _with_base_path(self, path: str) -> str:
        """Prefix relative app paths when PiKaraoke is mounted under a subpath."""
        return f"{self.base_path}{path}" if self.base_path else path

    def play_file(self, file_path: str) -> PlaybackResult:
        """Start playback of a media file.

        Handles file resolution, transcoding, and stream setup. Always
        transcodes: audio is always routed through rubberband so its pitch
        can be changed live later, which rules out a raw stream copy.

        Args:
            file_path: Path to the media file to play.

        Returns:
            PlaybackResult with success status and stream information.
        """
        from flask_babel import _

        streaming_format = self.streaming_format
        complete_transcode_before_play = self.preferences.get_or_default(
            "complete_transcode_before_play"
        )

        is_hls = streaming_format == "hls"

        try:
            fr = FileResolver(file_path, streaming_format)
        except Exception as e:
            error_message = _("Error resolving file: %s") % str(e)
            logging.error(error_message)
            return PlaybackResult(success=False, error=error_message)

        # Set stream URL based on format
        if is_hls:
            stream_url_path = self._with_base_path(f"/stream/{fr.stream_uid}.m3u8")
        elif complete_transcode_before_play:
            stream_url_path = self._with_base_path(f"/stream/full/{fr.stream_uid}")
        else:
            stream_url_path = self._with_base_path(f"/stream/{fr.stream_uid}.mp4")

        is_transcoding_complete, is_buffering_complete = self._transcode_file(fr, is_hls)

        subtitle_url = None
        if fr.ass_file_path:
            subtitle_url = self._with_base_path(f"/subtitle/{fr.stream_uid}")
            logging.debug(f"Subtitle file found: {fr.ass_file_path}. URL: {subtitle_url}")

        # Check if the stream is ready to play
        if is_transcoding_complete or is_buffering_complete:
            logging.debug("Stream ready!")
            return PlaybackResult(
                success=True,
                stream_url=stream_url_path,
                subtitle_url=subtitle_url,
                duration=fr.duration,
            )
        else:
            error_message = _("Failed to prepare stream")
            logging.error(error_message)
            return PlaybackResult(success=False, error=error_message)

    def set_pitch(self, semitones: int) -> bool:
        """Change the pitch of the currently playing audio live, via zmq.

        Sends a runtime command to the running ffmpeg process's rubberband
        filter instead of restarting it, so playback is uninterrupted.

        Args:
            semitones: Number of semitones to transpose (0 = original key).

        Returns:
            True if the command was sent and acknowledged, False otherwise
            (e.g. nothing playing, or ffmpeg's command socket didn't respond).
        """
        if self.ffmpeg_process is None or self.pitch_control_port is None:
            logging.warning("Cannot change pitch: no song currently playing")
            return False

        pitch_factor = 2 ** (semitones / 12)
        socket_ = self._zmq_context.socket(zmq.REQ)
        try:
            socket_.setsockopt(zmq.LINGER, 0)
            socket_.setsockopt(zmq.RCVTIMEO, 2000)
            socket_.setsockopt(zmq.SNDTIMEO, 2000)
            socket_.connect(f"tcp://127.0.0.1:{self.pitch_control_port}")
            socket_.send_string(f"rubberband pitch {pitch_factor}")
            reply = socket_.recv_string()
        except zmq.error.ZMQError as e:
            logging.error(f"Failed to send live pitch command: {e}")
            return False
        finally:
            socket_.close()

        if not reply.startswith("0 "):
            logging.error(f"FFmpeg rejected pitch command: {reply}")
            return False
        return True

    def _transcode_file(self, fr: FileResolver, is_hls: bool) -> tuple[bool, bool]:
        """Transcode a file using FFmpeg.

        Args:
            fr: FileResolver instance with file information.
            is_hls: Whether to use HLS streaming format.

        Returns:
            Tuple of (is_transcoding_complete, is_buffering_complete).
        """
        self.kill_ffmpeg()

        normalize_audio = self.preferences.get_or_default("normalize_audio")
        complete_transcode_before_play = self.preferences.get_or_default(
            "complete_transcode_before_play"
        )
        avsync = self.preferences.get_or_default("avsync")
        cdg_pixel_scaling = self.preferences.get_or_default("cdg_pixel_scaling")
        buffer_size = int(self.preferences.get_or_default("buffer_size")) * 1000

        self.pitch_control_port = _find_free_port()
        ffmpeg_args = build_ffmpeg_cmd(
            fr,
            self.pitch_control_port,
            normalize_audio,
            not is_hls,  # force mp4 encoding
            complete_transcode_before_play,
            avsync,
            cdg_pixel_scaling,
        )
        self.ffmpeg_process = subprocess.Popen(
            ["ffmpeg"] + ffmpeg_args, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._exit_logged_for = None

        # FFmpeg outputs to stderr - prevent blocking reads
        self.ffmpeg_log = Queue()
        t = Thread(
            target=enqueue_output,
            args=(self.ffmpeg_process.stderr, self.ffmpeg_log),
            daemon=True,
        )
        t.start()

        transcode_max_retries = 2500  # ~2 minutes max
        is_transcoding_complete = False
        is_buffering_complete = False

        # Transcoding readiness polling loop
        while True:
            self.log_ffmpeg_output()

            # Check if FFmpeg has exited
            if self.ffmpeg_process.poll() is not None:
                exitcode = self.ffmpeg_process.poll()
                if exitcode != 0:
                    logging.error(f"FFmpeg exited with code {exitcode}")
                    break
                else:
                    is_transcoding_complete = True
                    stream_size = fr.get_current_stream_size()
                    logging.debug(f"Transcoding complete. Output size: {stream_size}")
                    break

            # Check buffering progress based on streaming format
            if is_hls:
                is_buffering_complete = self._check_hls_buffer(fr, buffer_size)
            else:
                is_buffering_complete = self._check_mp4_buffer(fr, buffer_size)

            if is_buffering_complete:
                break

            # Prevent infinite loop
            if transcode_max_retries <= 0:
                logging.error("Max retries reached trying to play song")
                break
            transcode_max_retries -= 1
            time.sleep(0.05)

        return is_transcoding_complete, is_buffering_complete

    def _check_hls_buffer(self, fr: FileResolver, buffer_size: int) -> bool:
        """Check if HLS buffer is ready for playback.

        Counts segment files directly instead of parsing the playlist.
        This works with hls_playlist_type=vod (playlist written at end)
        while still allowing early playback detection.

        Args:
            fr: FileResolver instance.
            buffer_size: Minimum buffer size in bytes.

        Returns:
            True if buffer is ready, False otherwise.
        """
        complete_transcode_before_play = self.preferences.get_or_default(
            "complete_transcode_before_play"
        )
        if complete_transcode_before_play:
            return False

        try:
            # Check if the playlist exists and has content
            if not os.path.exists(fr.output_file):
                return False
            if os.path.getsize(fr.output_file) == 0:
                return False

            # Count segment files directly (works even before playlist is written)
            stream_uid_str = str(fr.stream_uid)
            segment_files = [
                f for f in os.listdir(fr.tmp_dir) if stream_uid_str in f and f.endswith(".m4s")
            ]
            segment_count = len(segment_files)
            min_segments = 3

            if segment_count >= min_segments:
                stream_size = fr.get_current_stream_size()
                if stream_size >= buffer_size:
                    logging.debug(
                        f"Buffering complete. Stream size: {stream_size}, "
                        f"Segments: {segment_count}"
                    )
                    return True
        except FileNotFoundError:
            pass  # Temp dir doesn't exist yet
        except OSError as e:
            logging.warning(f"I/O error checking buffer: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during buffering check: {e}")

        return False

    def _check_mp4_buffer(self, fr: FileResolver, buffer_size: int) -> bool:
        """Check if MP4 buffer is ready for playback.

        Args:
            fr: FileResolver instance.
            buffer_size: Minimum buffer size in bytes.

        Returns:
            True if buffer is ready, False otherwise.
        """
        complete_transcode_before_play = self.preferences.get_or_default(
            "complete_transcode_before_play"
        )
        if complete_transcode_before_play:
            return False

        try:
            output_file_size = os.path.getsize(fr.output_file)
            if output_file_size > buffer_size:
                logging.debug(f"Buffering complete. File size: {output_file_size}")
                return True
        except FileNotFoundError:
            pass

        return False

    def log_ffmpeg_output(self) -> None:
        """Log any pending FFmpeg output from the queue, and surface a crash.

        Buffering readiness is only checked once, while starting playback;
        once the stream is handed to the client nothing else watches the
        process, so a later crash would otherwise run the clock out on the
        client's own stall detector with no trace of why in the log.
        """
        if self.ffmpeg_log is None:
            return
        recent_lines: list[str] = []
        while self.ffmpeg_log.qsize() > 0:
            output = self.ffmpeg_log.get_nowait()
            line = output.decode("utf-8", "ignore").strip()
            logging.debug("[FFMPEG] " + line)
            recent_lines.append(line)

        process = self.ffmpeg_process
        exit_code = process.poll() if process else None
        if (
            process is not None
            and process is not self._exit_logged_for
            and exit_code
            not in (
                None,
                0,
            )
        ):
            self._exit_logged_for = process
            tail = "\n".join(recent_lines[-20:])
            logging.error(f"FFmpeg exited unexpectedly with code {exit_code}:\n{tail}")

    def kill_ffmpeg(self) -> None:
        """Terminate the running FFmpeg process gracefully.

        Uses SIGTERM first, then SIGKILL if needed.
        Critical for Raspberry Pi to release GPU memory from h264_v4l2m2m encoder.
        """
        if self.ffmpeg_process:
            logging.debug("Terminating ffmpeg process gracefully")
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
                logging.debug("FFmpeg process terminated gracefully")
            except subprocess.TimeoutExpired:
                logging.warning("FFmpeg did not terminate gracefully, forcing kill")
                self.ffmpeg_process.kill()
                self.ffmpeg_process.wait()
                logging.debug("FFmpeg process force killed")
            except Exception as e:
                logging.debug(f"FFmpeg termination exception: {e}")
            finally:
                self.ffmpeg_process = None
                self.pitch_control_port = None
