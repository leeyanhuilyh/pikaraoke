"""Demucs-backed vocal separation with an on-disk cache.

Separation is the most expensive operation in the app - minutes of CPU for a
single song - so the separated track is cached and reused for every later play
of that song, and demucs never runs twice on the same source.

Only the vocals-removed track is kept. The original file already is the
"vocals on" side of the toggle, so the isolated vocal stem demucs also produces
has no playback use and is discarded.
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
import threading
from queue import Queue

from gevent import Greenlet, spawn

from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.get_platform import use_spare_capacity
from pikaraoke.lib.metadata_parser import extract_youtube_id
from pikaraoke.lib.preference_manager import PreferenceManager
from pikaraoke.lib.song_list import STEMS_DIR_NAME

# Separation modes, as stored in the vocal_separation preference.
OFF = "off"
BACKGROUND = "background"
BEFORE_PLAY = "before_play"
MODES = (OFF, BACKGROUND, BEFORE_PLAY)

# Let demucs pick the device itself, rather than pinning one.
DEVICE_AUTO = "auto"

# Hybrid Transformer Demucs: the fork's default model, and the best quality of
# the bundled models for the 2-stem split we need.
MODEL = "htdemucs"

# The separated track is only ever a backing track behind a live singer, so it
# does not need the 320k the demucs CLI defaults to.
MP3_BITRATE = "192k"

_NO_VOCALS_SUFFIX = ".no_vocals.mp3"
_FINGERPRINT_SUFFIX = ".json"


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

    Runs separation either inline (blocking the caller) or through a serial
    background queue, depending on the vocal_separation preference. The queue
    is serial because a single separation already saturates the CPU.
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
        self._queue: Queue = Queue()
        self._worker: Greenlet | None = None
        # One separation saturates the machine, and the background worker can
        # be partway through a song that playback has just come up to.
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background separation worker.

        A greenlet on the main gevent hub rather than an OS thread, for the same
        reason the download worker is one: the primitives it touches (Queue,
        subprocess) are monkey-patched.
        """
        self._worker = spawn(self._process_queue)
        logging.debug("Vocal separation worker started")

    @property
    def mode(self) -> str:
        """Current separation mode, falling back to off for an unknown value."""
        mode = self._preferences.get_or_default("vocal_separation")
        return mode if mode in MODES else OFF

    def is_available(self) -> bool:
        """Whether the vendored demucs submodule is present to run at all.

        Only reports that the code is there. Its heavy dependencies (torch)
        are installed separately, and a missing one surfaces as a failed run.
        """
        return _vendored_demucs_dir() is not None

    def cached_track(self, song_path: str) -> str | None:
        """Path to this song's cached vocals-removed track, if one is current.

        A cached track whose source has changed size or mtime since it was made
        belongs to a file that is no longer there, so it is reported as a miss.
        """
        key = _cache_key(self._songs_dir, song_path)
        track = os.path.join(stems_dir(self._songs_dir), key + _NO_VOCALS_SUFFIX)
        if not os.path.isfile(track):
            return None

        fingerprint_path = os.path.join(stems_dir(self._songs_dir), key + _FINGERPRINT_SUFFIX)
        try:
            with open(fingerprint_path, encoding="utf-8") as f:
                recorded = json.load(f)
        except (OSError, ValueError) as e:
            logging.debug(f"Unreadable stem fingerprint for {song_path}: {e}")
            return None

        if recorded != _fingerprint(song_path):
            logging.info(f"Cached vocal track is stale, will re-separate: {song_path}")
            return None
        return track

    def remove_cached(self, song_path: str) -> None:
        """Delete a song's cached track, so deleting a song reclaims its space."""
        key = _cache_key(self._songs_dir, song_path)
        directory = stems_dir(self._songs_dir)
        for suffix in (_NO_VOCALS_SUFFIX, _FINGERPRINT_SUFFIX):
            path = os.path.join(directory, key + suffix)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as e:
                logging.warning(f"Could not remove cached stem {path}: {e}")

    def queue_separation(self, song_path: str) -> None:
        """Hand a song to the background worker, unless it is already cached."""
        if self.cached_track(song_path):
            return
        self._queue.put(song_path)

    def separate(self, song_path: str) -> str | None:
        """Separate a song now, blocking until it finishes.

        Separations run one at a time. A caller arriving while the same song is
        already being separated waits for it and gets its cached result rather
        than running demucs a second time.

        Returns:
            Path to the cached vocals-removed track, or None if separation
            could not be run or did not produce one.
        """
        with self._lock:
            cached = self.cached_track(song_path)
            if cached:
                return cached
            return self._separate_uncached(song_path)

    def _separate_uncached(self, song_path: str) -> str | None:
        """Run demucs for a song known not to be cached. Caller holds the lock."""
        demucs_dir = _vendored_demucs_dir()
        if demucs_dir is None:
            logging.error(
                "Vocal separation is enabled but the demucs submodule is missing. "
                "Run: git submodule update --init"
            )
            return None

        # Recorded before separation: a song replaced while demucs was running
        # would otherwise be fingerprinted as the source of a track made from
        # the file it replaced.
        fingerprint = _fingerprint(song_path)
        if fingerprint is None:
            return None

        with tempfile.TemporaryDirectory(prefix="pikaraoke-demucs-") as work_dir:
            if not self._run_demucs(song_path, work_dir, demucs_dir):
                return None
            produced = _find_no_vocals(work_dir)
            if produced is None:
                logging.error(f"Demucs produced no vocals-removed track for {song_path}")
                return None
            encoded = os.path.join(work_dir, "no_vocals.mp3")
            if not _encode_mp3(produced, encoded):
                return None
            return self._store(song_path, encoded, fingerprint)

    def _run_demucs(self, song_path: str, out_dir: str, demucs_dir: str) -> bool:
        """Run the demucs CLI over one song. Returns True on a clean exit."""
        cmd = [
            sys.executable,
            "-m",
            "demucs.separate",
            "--two-stems",
            "vocals",
            "-n",
            MODEL,
            "-o",
            out_dir,
        ]
        device = self._preferences.get_or_default("vocal_separation_device")
        if device and device != DEVICE_AUTO:
            cmd += ["-d", device]
        cmd.append(song_path)
        # The vendored fork is run in place rather than installed, so its own
        # directory goes on the path ahead of any demucs in site-packages.
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [demucs_dir, env["PYTHONPATH"]] if env.get("PYTHONPATH") else [demucs_dir]
        )

        logging.info(f"Separating vocals: {song_path}")
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        except OSError as e:
            logging.error(f"Could not start demucs: {e}")
            return False

        use_spare_capacity(process)
        output, _unused = process.communicate()
        if process.returncode != 0:
            logging.error(f"Demucs failed for {song_path} (exit {process.returncode}): {output}")
            return False
        return True

    def _store(self, song_path: str, produced: str, fingerprint: dict) -> str | None:
        """Move a finished track into the cache and record what it was made from."""
        directory = stems_dir(self._songs_dir)
        key = _cache_key(self._songs_dir, song_path)
        track = os.path.join(directory, key + _NO_VOCALS_SUFFIX)
        try:
            os.makedirs(directory, exist_ok=True)
            # Across filesystems when the temp dir and the library differ.
            shutil.move(produced, track)
            with open(
                os.path.join(directory, key + _FINGERPRINT_SUFFIX), "w", encoding="utf-8"
            ) as f:
                json.dump(fingerprint, f)
        except OSError as e:
            logging.error(f"Could not cache separated track for {song_path}: {e}")
            return None

        logging.info(f"Vocal separation complete: {song_path}")
        self._events.emit("vocals_separated", song_path, track)
        return track

    def _process_queue(self) -> None:
        """Serially separate queued songs, forever."""
        while True:
            song_path = self._queue.get()
            try:
                self.separate(song_path)
            except Exception as e:
                # Deliberately broad: one song that cannot be separated must not
                # take the worker down and strand every song queued behind it.
                logging.error(f"Error separating {song_path}: {e}")
            finally:
                self._queue.task_done()


def _vendored_demucs_dir() -> str | None:
    """Directory of the vendored demucs fork, or None when it is not checked out."""
    # This file is <repo>/pikaraoke/lib/vocal_separator.py
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(repo_root, "vendor", "demucs")
    return path if os.path.isdir(os.path.join(path, "demucs")) else None


def _encode_mp3(wav_path: str, mp3_path: str) -> bool:
    """Encode demucs' lossless output to mp3 with ffmpeg, not demucs' own encoder.

    Demucs' built-in mp3 encoder writes no gapless header, so the encoder's
    1105-sample delay is never trimmed and the track plays about 25ms late
    against the original. ffmpeg records the delay in the file, and trims it
    again when it decodes, so the length and alignment match the source.
    """
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", wav_path, "-c:a", "libmp3lame", "-b:a", MP3_BITRATE]
    try:
        result = subprocess.run(cmd + [mp3_path], capture_output=True, text=True, check=False)
    except OSError as e:
        logging.error(f"Could not start ffmpeg to encode the separated track: {e}")
        return False
    if result.returncode != 0:
        logging.error(f"ffmpeg failed encoding the separated track: {result.stderr}")
        return False
    return True


def _find_no_vocals(work_dir: str) -> str | None:
    """Locate the no_vocals file demucs wrote under its model-named subfolder."""
    for dirpath, _dirnames, filenames in os.walk(work_dir):
        for filename in filenames:
            if filename.startswith("no_vocals."):
                return os.path.join(dirpath, filename)
    return None
