"""Vocal separation with an on-disk cache.

Separation is the most expensive operation in the app - a minute or more of CPU
for a single song - so the separated track is cached and reused for every later
play of that song, and a song is never separated twice.

The track is written as it is produced, so a song can start playing, and its
vocals-off track can follow along just behind, long before the whole song is
done. Only the vocals-removed track is kept: the original file already is the
"vocals on" side of the toggle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

from gevent import Greenlet, sleep, spawn
from gevent.event import Event

from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.ffmpeg import (
    PCM_BYTES_PER_SECOND,
    PCM_CHANNELS,
    PCM_FORMAT,
    PCM_SAMPLE_RATE,
)
from pikaraoke.lib.get_platform import get_data_directory, use_spare_capacity
from pikaraoke.lib.metadata_parser import extract_youtube_id
from pikaraoke.lib.preference_manager import PreferenceManager
from pikaraoke.lib.song_list import STEMS_DIR_NAME

# Separation modes, as stored in the vocal_separation preference.
OFF = "off"
BACKGROUND = "background"
BEFORE_PLAY = "before_play"
MODES = (OFF, BACKGROUND, BEFORE_PLAY)

# The separated track is only ever a backing track behind a live singer.
MP3_BITRATE = "192k"

# How much vocals-removed audio a song needs before it starts, in before_play
# mode. Separation then runs faster than playback, so it stays ahead from here.
PLAYBACK_HEAD_START_SECONDS = 15
HEAD_START_TIMEOUT_SECONDS = 180

_NO_VOCALS_SUFFIX = ".no_vocals.mp3"
_FINGERPRINT_SUFFIX = ".json"
# Raw PCM written while a separation runs, encoded to the mp3 once it finishes.
_PARTIAL_SUFFIX = ".partial.pcm"


def stems_dir(songs_dir: str) -> str:
    """Directory holding cached stems for a songs library."""
    return os.path.join(songs_dir, STEMS_DIR_NAME)


def _cache_key(songs_dir: str, song_path: str) -> str:
    """Stable identity for a song's cache entry.

    Keyed on the YouTube ID where there is one, so renaming a downloaded song
    (including by the batch renamer) keeps its cached track. Songs without an
    ID fall back to their path, and so do lose the cache on a rename; the
    fingerprint check just treats that as a miss and separates again.
    """
    youtube_id = extract_youtube_id(song_path)
    if youtube_id:
        return youtube_id
    relative = os.path.relpath(song_path, songs_dir)
    return hashlib.sha1(relative.encode("utf-8", "replace")).hexdigest()[:16]


def _fingerprint(song_path: str) -> dict | None:
    """Size and mtime of the source, to detect a song replaced after separation."""
    try:
        stat = os.stat(song_path)
    except OSError as e:
        logging.debug(f"Could not stat {song_path}: {e}")
        return None
    return {"size": stat.st_size, "mtime": int(stat.st_mtime)}


class VocalSeparator:
    """Produces and caches a vocals-removed track for a song.

    A background worker separates songs one at a time, since one separation
    already saturates the CPU, in the order they were handed over, except that
    the song about to play always jumps the line (see prioritize).
    """

    def __init__(
        self,
        events: EventSystem,
        preferences: PreferenceManager,
        songs_dir: str,
    ) -> None:
        self._events = events
        self._preferences = preferences
        self._songs_dir = songs_dir
        self._pending: list[str] = []
        self._wake = Event()
        self._worker: Greenlet | None = None
        self._current: str | None = None
        self._process: subprocess.Popen | None = None
        self._interrupted = False

    def start(self) -> None:
        """Start the background separation worker.

        A greenlet on the main gevent hub rather than an OS thread, for the same
        reason the download worker is one: the primitives it touches (Event,
        subprocess) are monkey-patched.
        """
        self._sweep_partials()
        self._worker = spawn(self._process_queue)
        logging.debug("Vocal separation worker started")

    @property
    def mode(self) -> str:
        """Current separation mode, falling back to off for an unknown value."""
        mode = self._preferences.get_or_default("vocal_separation")
        return mode if mode in MODES else OFF

    def _path(self, song_path: str, suffix: str) -> str:
        return os.path.join(
            stems_dir(self._songs_dir), _cache_key(self._songs_dir, song_path) + suffix
        )

    def cached_track(self, song_path: str) -> str | None:
        """Path to this song's cached vocals-removed track, if one is current.

        A cached track whose source has changed size or mtime since it was made
        belongs to a file that is no longer there, so it is reported as a miss.
        """
        track = self._path(song_path, _NO_VOCALS_SUFFIX)
        if not os.path.isfile(track):
            return None
        try:
            with open(self._path(song_path, _FINGERPRINT_SUFFIX), encoding="utf-8") as f:
                recorded = json.load(f)
        except (OSError, ValueError) as e:
            logging.debug(f"Unreadable stem fingerprint for {song_path}: {e}")
            return None

        if recorded != _fingerprint(song_path):
            logging.info(f"Cached vocal track is stale, will re-separate: {song_path}")
            return None
        return track

    def partial_track(self, song_path: str) -> str | None:
        """The raw track being written right now for this song, if it is the one separating.

        It only ever grows, and stops growing once this returns None for it.
        """
        return self._path(song_path, _PARTIAL_SUFFIX) if song_path == self._current else None

    def seconds_ready(self, song_path: str) -> float:
        """How many seconds of this song's vocals-removed track exist so far."""
        if self.cached_track(song_path):
            return float("inf")
        partial = self.partial_track(song_path)
        if partial is None:
            return 0.0
        try:
            return os.path.getsize(partial) / PCM_BYTES_PER_SECOND
        except OSError:
            return 0.0

    def remove_cached(self, song_path: str) -> None:
        """Delete a song's cached track, so deleting a song reclaims its space."""
        for suffix in (_NO_VOCALS_SUFFIX, _FINGERPRINT_SUFFIX, _PARTIAL_SUFFIX):
            path = self._path(song_path, suffix)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as e:
                logging.warning(f"Could not remove cached stem {path}: {e}")

    def queue_separation(self, song_path: str) -> None:
        """Hand a song to the background worker, unless it is cached or already in line."""
        if song_path == self._current or song_path in self._pending:
            return
        if self.cached_track(song_path):
            return
        self._pending.append(song_path)
        self._wake.set()

    def prioritize(self, song_path: str) -> None:
        """Separate this song next, ahead of the queue and of whatever is separating now.

        For the song about to play: its vocals-off track has to stay ahead of
        playback, so it can't wait for another song to finish first. A song it
        interrupts goes back in line right behind it, and starts over.
        """
        if song_path == self._current or self.cached_track(song_path):
            return
        if song_path in self._pending:
            self._pending.remove(song_path)
        self._pending.insert(0, song_path)
        if self._current is not None and self._process is not None:
            logging.info(f"Pausing vocal separation of {self._current} for the song about to play")
            self._pending.insert(1, self._current)
            self._interrupted = True
            self._process.terminate()
        self._wake.set()

    def wait_for_head_start(
        self,
        song_path: str,
        seconds: float = PLAYBACK_HEAD_START_SECONDS,
        timeout: float = HEAD_START_TIMEOUT_SECONDS,
    ) -> bool:
        """Block until `seconds` of the song's vocals-removed track exist.

        Returns False if separation fails, or doesn't get there within `timeout`.
        """
        self.prioritize(song_path)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.seconds_ready(song_path) >= seconds:
                return True
            if song_path != self._current and song_path not in self._pending:
                return False  # it was separated and nothing was cached: it failed
            sleep(0.2)
        logging.warning(f"Vocal separation had no head start after {timeout:.0f}s: {song_path}")
        return False

    def _process_queue(self) -> None:
        """Separate queued songs one at a time, forever."""
        while True:
            if not self._pending:
                self._wake.clear()
                self._wake.wait()
                continue
            self._separate_next()

    def _separate_next(self) -> None:
        """Separate the song at the front of the line, unless it is cached by now."""
        song_path = self._pending.pop(0)
        if self.cached_track(song_path):
            return
        self._current = song_path
        try:
            self._separate(song_path)
        except Exception as e:
            # Deliberately broad: one song that cannot be separated must not
            # take the worker down and strand every song queued behind it.
            logging.error(f"Error separating {song_path}: {e}")
        finally:
            # Cleared only now, after the track is cached, so a reader of the
            # partial track never sees it stop growing before it's finished.
            self._current = None
            self._process = None
            self._interrupted = False
            self._sweep_partials()

    def _separate(self, song_path: str) -> None:
        """Separate one song, writing its partial track as it goes, then cache it."""
        # Recorded before separation: a song replaced while it was separating
        # would otherwise be fingerprinted as the source of a track made from
        # the file it replaced.
        fingerprint = _fingerprint(song_path)
        if fingerprint is None:
            return
        partial = self._path(song_path, _PARTIAL_SUFFIX)
        os.makedirs(stems_dir(self._songs_dir), exist_ok=True)

        cmd = [sys.executable, "-m", "pikaraoke.lib.mdx", song_path, partial]
        cmd += ["--model-dir", os.path.join(get_data_directory(), "models")]
        logging.info(f"Separating vocals: {song_path}")
        try:
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
        except OSError as e:
            logging.error(f"Could not start the vocal separator: {e}")
            return
        use_spare_capacity(self._process)
        self._events.emit("vocals_separating", song_path)

        output, _unused = self._process.communicate()
        if self._interrupted:
            return
        if self._process.returncode != 0:
            logging.error(
                f"Vocal separation failed for {song_path} "
                f"(exit {self._process.returncode}): {output}"
            )
            return

        with tempfile.TemporaryDirectory(prefix="pikaraoke-separation-") as work_dir:
            encoded = os.path.join(work_dir, "no_vocals.mp3")
            if _encode_mp3(partial, encoded):
                self._store(song_path, encoded, fingerprint)

    def _sweep_partials(self) -> None:
        """Delete partial tracks no longer being written.

        Those are finished separations, cached by now, and interrupted ones,
        which start over. One a player still has open can't be deleted on
        Windows, so it is left for the next sweep.
        """
        directory = stems_dir(self._songs_dir)
        try:
            names = os.listdir(directory)
        except OSError:
            return
        writing = self.partial_track(self._current) if self._current else None
        for name in names:
            path = os.path.join(directory, name)
            if name.endswith(_PARTIAL_SUFFIX) and path != writing:
                try:
                    os.remove(path)
                except OSError as e:
                    logging.debug(f"Partial track still in use, leaving it for later: {e}")

    def _store(self, song_path: str, produced: str, fingerprint: dict) -> str | None:
        """Move a finished track into the cache and record what it was made from."""
        directory = stems_dir(self._songs_dir)
        track = self._path(song_path, _NO_VOCALS_SUFFIX)
        try:
            os.makedirs(directory, exist_ok=True)
            # Across filesystems when the temp dir and the library differ.
            shutil.move(produced, track)
            with open(self._path(song_path, _FINGERPRINT_SUFFIX), "w", encoding="utf-8") as f:
                json.dump(fingerprint, f)
        except OSError as e:
            logging.error(f"Could not cache separated track for {song_path}: {e}")
            return None

        logging.info(f"Vocal separation complete: {song_path}")
        self._events.emit("vocals_separated", song_path, track)
        return track


def _encode_mp3(pcm_path: str, mp3_path: str) -> bool:
    """Encode the separator's raw PCM to mp3 with ffmpeg.

    ffmpeg records the mp3 encoder's 1105-sample delay in the file and trims it
    again when decoding, so the track stays sample-aligned with the original.
    An encoder that skips that header leaves the track about 25ms late.
    """
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", PCM_FORMAT, "-ar", str(PCM_SAMPLE_RATE)]
    cmd += ["-ac", str(PCM_CHANNELS)]
    cmd += ["-i", pcm_path, "-c:a", "libmp3lame", "-b:a", MP3_BITRATE, mp3_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as e:
        logging.error(f"Could not start ffmpeg to encode the separated track: {e}")
        return False
    if result.returncode != 0:
        logging.error(f"ffmpeg failed encoding the separated track: {result.stderr}")
        return False
    return True
