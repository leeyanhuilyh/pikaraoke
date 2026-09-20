"""Playback controller for managing video playback state and coordination."""

import logging
import os
import time
from typing import TYPE_CHECKING, Callable

from flask_babel import _

from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.file_resolver import FileResolver, delete_tmp_dir
from pikaraoke.lib.preference_manager import PreferenceManager
from pikaraoke.lib.rendition_manager import RenditionManager
from pikaraoke.lib.stream_manager import PlaybackResult, StreamManager
from pikaraoke.lib.vocal_separator import OFF

if TYPE_CHECKING:
    import subprocess

    from pikaraoke.lib.vocal_separator import VocalSeparator


class PlaybackController:
    """Controller for managing playback state and stream coordination.

    Owns all "now playing" state and coordinates with StreamManager for
    FFmpeg transcoding and playback.

    Attributes:
        now_playing: Title of the currently playing song.
        now_playing_filename: File path of the currently playing song.
        now_playing_user: User who queued the current song.
        now_playing_transpose: Semitones to transpose current song.
        now_playing_vocals: Whether the current song is playing with its vocals.
        now_playing_duration: Duration of current song in seconds.
        now_playing_url: Stream URL for current song.
        now_playing_subtitle_url: URL path for subtitles.
        now_playing_position: Current playback position in seconds.
        is_paused: Whether playback is paused.
        is_playing: Whether a song is currently playing.
        ffmpeg_process: Currently running FFmpeg subprocess.
    """

    now_playing: str | None = None
    now_playing_filename: str | None = None
    now_playing_user: str | None = None
    now_playing_transpose: int = 0
    now_playing_vocals: bool = True
    now_playing_duration: int | None = None
    now_playing_url: str | None = None
    now_playing_subtitle_url: str | None = None
    now_playing_position: float | None = None
    is_paused: bool = True
    is_playing: bool = False

    def __init__(
        self,
        preferences: PreferenceManager,
        events: EventSystem,
        filename_from_path: Callable[[str, bool], str],
        streaming_format: str = "hls",
        base_path: str = "",
        is_transpose_enabled: bool = False,
        vocal_separator: "VocalSeparator | None" = None,
    ) -> None:
        """Initialize the playback controller.

        Args:
            preferences: PreferenceManager instance for configuration.
            events: EventSystem instance for event emission.
            filename_from_path: Function to extract display name from path.
            streaming_format: Video streaming format ('hls' or 'mp4').
            base_path: URL path prefix when PiKaraoke is hosted under a subpath.
            is_transpose_enabled: Whether ffmpeg has the rubberband filter,
                gating whether pitch pre-rendering can be used at all.
            vocal_separator: Source of each song's vocals-removed track, which
                is what makes switching vocals off possible.
        """
        self.preferences = preferences
        self.events = events
        self.filename_from_path = filename_from_path
        self.is_transpose_enabled = is_transpose_enabled
        self.stream_manager = StreamManager(preferences, streaming_format, base_path)
        self._vocal_separator = vocal_separator
        self.rendition_manager = RenditionManager(preferences, base_path, vocal_separator)
        self._using_rendition_manager = False

    @property
    def ffmpeg_process(self) -> "subprocess.Popen | None":
        """Get the current FFmpeg process."""
        return self.stream_manager.ffmpeg_process

    def play_file(self, file_path: str, user: str, semitones: int = 0) -> PlaybackResult:
        """Start playback of a media file.

        Blocks until client connects or timeout occurs.

        Args:
            file_path: Path to the media file to play.
            user: User who queued the song.
            semitones: Number of semitones to transpose (0 = no change).

        Returns:
            PlaybackResult with success status and stream information.
        """
        if not os.path.isfile(file_path):
            error_msg = _("Song file not found: %s") % file_path
            logging.warning(error_msg)
            self.now_playing_filename = None
            return PlaybackResult(success=False, error=error_msg)

        logging.info(
            f"Playing file: {file_path} for user: {user}, transposed {semitones} semitones"
        )

        self.claim(file_path)

        # Switching vocals needs the same alternate-audio machinery as switching
        # pitch, so either feature being on is enough to use it.
        pitch_switching = (
            self.is_transpose_enabled
            and int(self.preferences.get_or_default("pitch_window_semitones")) > 0
        )
        vocal_switching = self._vocal_separator is not None and self._vocal_separator.mode != OFF
        use_rendition_manager = self.stream_manager.streaming_format == "hls" and (
            pitch_switching or vocal_switching
        )
        if use_rendition_manager:
            try:
                fr = FileResolver(file_path, self.stream_manager.streaming_format)
            except Exception as e:
                error_message = _("Error resolving file: %s") % str(e)
                logging.error(error_message)
                result = PlaybackResult(success=False, error=error_message)
            else:
                result = self.rendition_manager.start(fr, semitones, file_path)
            self._using_rendition_manager = True
        else:
            result = self.stream_manager.play_file(file_path, semitones)
            self._using_rendition_manager = False

        if not result.success:
            self.now_playing_filename = None
            return result

        self.now_playing = self.filename_from_path(file_path, remove_youtube_id=True)
        self.now_playing_user = user
        self.now_playing_transpose = semitones
        self.now_playing_vocals = True
        self.now_playing_duration = result.duration
        self.now_playing_url = result.stream_url
        self.now_playing_subtitle_url = result.subtitle_url
        self.is_paused = False

        self.events.emit("playback_started")

        # Wait for client to connect
        max_retries = 100
        while not self.is_playing and max_retries > 0:
            time.sleep(0.1)
            max_retries -= 1

        if not self.is_playing:
            error_msg = _("Stream was not playable! Skipping track")
            logging.error(error_msg)
            self.end_song(reason="timeout")
            return PlaybackResult(success=False, error=error_msg)

        logging.debug("Stream is playing")
        return result

    def claim(self, file_path: str) -> None:
        """Mark a file as the one playback owns, before any of it is read.

        This field is what tells a rename request the file is spoken for, so it
        must be set before the first yield point: play_file() sleeps through
        transcoding and gevent serves HTTP requests during those sleeps.
        """
        self.now_playing_filename = file_path

    def start_song(self) -> None:
        """Mark the current song as actively playing.

        Called by Flask route when client connects to stream.
        Idempotent - safe to call multiple times.
        """
        if not self.is_playing:
            logging.info(f"Song starting: {self.now_playing}")
            self.is_playing = True

    def end_song(self, reason: str | None = None) -> None:
        """End the current song and clean up resources.

        Args:
            reason: Optional reason for ending (e.g., 'complete', 'skip', 'timeout').
        """
        logging.info(f"Song ending: {self.now_playing}")
        if reason:
            logging.info(f"Reason: {reason}")
            if reason not in ("complete", "skip", "transpose"):
                # MSG: Message shown when the song ends abnormally
                self.events.emit("notification", _("Song ended abnormally: %s") % reason, "danger")

        self.reset_now_playing()
        if self._using_rendition_manager:
            self.rendition_manager.kill_all()
        else:
            self.stream_manager.kill_ffmpeg()
        # Small delay to ensure FFmpeg fully terminates and file handles close
        # Critical on Raspberry Pi with slow SD cards and hardware encoder cleanup
        time.sleep(0.3)
        delete_tmp_dir()
        logging.debug("Cleanup complete")

        self.events.emit("song_ended", reason)

    def skip(self, log_action: bool = True, reason: str = "skip") -> bool:
        """Skip the currently playing song.

        Args:
            log_action: Whether to log and notify about the skip.
            reason: End reason passed to listeners. Callers that end the stream
                without ending the performance (a transpose restarts the same
                song in a new key) pass their own, so play history can tell the
                difference between a real skip and a restart.

        Returns:
            True if a song was skipped, False if nothing playing.
        """
        if self.is_playing:
            if log_action:
                # MSG: Message shown after the song is skipped, will be followed by song name
                self.events.emit("notification", _("Skip: %s") % self.now_playing, "info")
            self.end_song(reason=reason)
            return True
        else:
            logging.warning("Tried to skip, but no file is playing!")
            return False

    def pause(self) -> bool:
        """Toggle pause state of the current song.

        Returns:
            True if successful, False if nothing playing.
        """
        if self.is_playing:
            if self.is_paused:
                # MSG: Message shown after the song is resumed, will be followed by song name
                self.events.emit("notification", _("Resume: %s") % self.now_playing, "info")
            else:
                # MSG: Message shown after the song is paused, will be followed by song name
                self.events.emit("notification", _("Pause: %s") % self.now_playing, "info")
            self.is_paused = not self.is_paused
            self.events.emit("now_playing_update")
            return True
        else:
            logging.warning("Tried to pause, but no file is playing!")
            return False

    def can_fast_switch(self, semitones: int) -> bool:
        """Whether a pitch change can switch HLS audio renditions instead of restarting.

        Args:
            semitones: Requested transpose value.

        Returns:
            True if pre-rendering is active for the current song and this
            pitch is one the master playlist declares. The rendition does
            not have to exist yet - it can be rendered on demand and
            switched to while it is still writing. Only a pitch outside the
            declared range needs a new playlist, and therefore a restart.
        """
        return self._using_rendition_manager and self.rendition_manager.is_switchable(
            semitones, self.now_playing_vocals
        )

    def can_switch_vocals(self, vocals_on: bool) -> bool:
        """Whether the current song can switch to this vocals state without restarting.

        Turning vocals off also needs the song's separated track to exist,
        which in background mode may still be a while after the song starts.
        """
        if not self._using_rendition_manager:
            return False
        if not self.rendition_manager.is_switchable(self.now_playing_transpose, vocals_on):
            return False
        return vocals_on or self.rendition_manager.no_vocals_available()

    def prepare_switch(self, semitones: int, vocals_on: bool) -> bool:
        """Get a rendition ready to switch to, rendering it next if needed.

        Passes the current playhead so the wait covers where playback
        actually is, not just the start of the rendition.
        """
        if not self._using_rendition_manager:
            return False
        return self.rendition_manager.ensure_switchable(
            semitones, vocals_on, position=self.now_playing_position or 0
        )

    def get_now_playing(self) -> dict[str, str | int | float | bool | None | list[int]]:
        """Get the current playback state.

        Returns:
            Dictionary with now playing information.
        """
        return {
            "now_playing": self.now_playing,
            "now_playing_user": self.now_playing_user,
            "now_playing_duration": self.now_playing_duration,
            "now_playing_transpose": self.now_playing_transpose,
            "now_playing_vocals": self.now_playing_vocals,
            "vocals_switchable": self.is_playing
            and self.can_switch_vocals(not self.now_playing_vocals),
            "now_playing_url": self.now_playing_url,
            "now_playing_subtitle_url": self.now_playing_subtitle_url,
            "now_playing_position": self.now_playing_position,
            "is_paused": self.is_paused,
        }

    def reset_now_playing(self) -> None:
        """Reset all now playing state to defaults."""
        self.now_playing = None
        self.now_playing_filename = None
        self.now_playing_user = None
        self.now_playing_url = None
        self.now_playing_subtitle_url = None
        self.is_paused = True
        self.is_playing = False
        self.now_playing_transpose = 0
        self.now_playing_vocals = True
        self.now_playing_duration = None
        self.now_playing_position = None

    def log_output(self) -> None:
        """Log any pending FFmpeg output."""
        self.stream_manager.log_ffmpeg_output()
