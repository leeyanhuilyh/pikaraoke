"""Background pre-rendering of alternate-pitch audio HLS renditions.

The currently playing song's video never changes with pitch, so it is
rendered once and shared. A window of nearby semitones is rendered as
separate audio-only HLS renditions in the background, all referenced from
one master playlist as alternate AUDIO renditions, so the browser can
switch pitch by picking a different audio track (near-instant, no video
interruption) instead of restarting playback.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING

from pikaraoke.lib.ffmpeg import (
    HLS_SEGMENT_SECONDS,
    build_audio_only_ffmpeg_cmd,
    build_video_only_ffmpeg_cmd,
)
from pikaraoke.lib.preference_manager import PreferenceManager
from pikaraoke.lib.process_priority import lower_priority, restore_priority
from pikaraoke.lib.stream_manager import PlaybackResult
from pikaraoke.lib.url_prefix import normalize_url_base_path

if TYPE_CHECKING:
    from pikaraoke.lib.file_resolver import FileResolver

# Every rendition in this range is declared in the master playlist, so any
# of them can be switched to without restarting - pre-rendering only
# decides how fast that switch feels. +/-6 is what commercial karaoke
# machines offer, and it already reaches every key: +6 and -6 are the same
# pitch class an octave apart, so a wider range only adds shift artifacts.
MIN_SEMITONES = -6
MAX_SEMITONES = 6
READY_POLL_INTERVAL_SECONDS = 0.05
READY_TIMEOUT_SECONDS = 120
MIN_READY_SEGMENTS = 3
# Headroom past the playhead a switch waits for, so the player has
# something buffered ahead and doesn't stall the moment it switches.
SWITCH_LOOKAHEAD_SECONDS = 6
SWITCH_WAIT_TIMEOUT_SECONDS = 15

# Option 1 of 3 under evaluation for how pre-rendering should be paced:
# True renders the whole window before the song starts, so song start is
# the direct measurement of how long that takes on real hardware. False
# only blocks on the base pitch and renders the rest in the background
# while the song plays (the original behavior). Flip and compare.
BLOCK_PLAYBACK_UNTIL_WINDOW_RENDERED = True

# Background renders (not the base pitch, not an on-demand one - those
# always run immediately regardless) are capped to this many alive at
# once. Audio-only rendering is confirmed single-threaded per process, and
# video uses the Pi's hardware encoder rather than CPU, so this is a cap
# on concurrent processes, not cores directly - 2 leaves real headroom on
# a 4-core Pi for the video pipeline, the web server, and anything else
# happening at the same time. Measured need: letting renders pile up
# unbounded made each successive one slower than the last, from ~1s to
# ~12s across a 13-pitch window, as they increasingly starved each other.
MAX_CONCURRENT_BACKGROUND_RENDERS = 2


def semitone_label(semitones: int) -> str:
    """Filename-safe token for a semitone value: p3 / m3 / p0.

    A trailing separator is added by callers before use as a segment-count
    marker, since "p1" would otherwise also match "p12"'s segment files.
    """
    return f"p{semitones}" if semitones >= 0 else f"m{-semitones}"


def video_playlist_path(fr: "FileResolver") -> str:
    return f"{fr.tmp_dir}/{fr.stream_uid}_video.m3u8"


def audio_playlist_path(fr: "FileResolver", semitones: int) -> str:
    return f"{fr.tmp_dir}/{fr.stream_uid}_audio_{semitone_label(semitones)}.m3u8"


def build_master_playlist(fr: "FileResolver", offsets: list[int], base_semitones: int) -> str:
    """Build the master HLS playlist declaring every windowed pitch rendition.

    Alternate audio renditions must all be listed up front: unlike a media
    playlist, a master playlist is loaded once by the client and never
    re-polled for entries added later.
    """
    lines = ["#EXTM3U", "#EXT-X-VERSION:7"]
    for s in offsets:
        default = "YES" if s == base_semitones else "NO"
        lines.append(
            f'#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="{s}",'
            f"DEFAULT={default},AUTOSELECT=YES,"
            f'URI="{fr.stream_uid}_audio_{semitone_label(s)}.m3u8"'
        )
    lines.append('#EXT-X-STREAM-INF:BANDWIDTH=5000000,CODECS="avc1.640028,mp4a.40.2",AUDIO="audio"')
    lines.append(f"{fr.stream_uid}_video.m3u8")
    return "\n".join(lines) + "\n"


def _terminate(proc: subprocess.Popen | None, timeout: float = 5) -> None:
    """Terminate a process gracefully, escalating to SIGKILL if it won't stop."""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    except Exception as e:
        logging.debug(f"Process termination exception: {e}")


def _count_segments(tmp_dir: str, marker: str) -> int:
    """Count finished segment files belonging to one rendition."""
    try:
        return len([f for f in os.listdir(tmp_dir) if marker in f and f.endswith(".m4s")])
    except OSError:
        return 0


def _wait_until_ready(
    tmp_dir: str, marker: str, playlist_path: str, proc: subprocess.Popen, stop_event: Event
) -> bool:
    """Poll until a rendition is safely playable.

    Either it has enough segments on disk to start streaming, or ffmpeg has
    already finished producing it - segment durations aren't guaranteed to
    land exactly on hls_time, so short content (or a short leftover tail)
    can finish with fewer than MIN_READY_SEGMENTS while still being
    complete and fully playable.
    """
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            if exit_code != 0:
                # A nonzero exit while we're mid-shutdown is us killing this
                # process on purpose (e.g. a restart superseded it), not a
                # real failure - don't log it as one.
                if not stop_event.is_set():
                    logging.error(f"FFmpeg exited with code {exit_code} rendering {marker}")
                return False
            return os.path.exists(playlist_path)
        if os.path.exists(playlist_path):
            if _count_segments(tmp_dir, marker) >= MIN_READY_SEGMENTS:
                return True
        time.sleep(READY_POLL_INTERVAL_SECONDS)
    return False


class RenditionManager:
    """Renders one shared video track plus a window of pitch-shifted audio
    tracks for the currently playing song, so an in-window pitch change can
    switch HLS audio renditions instead of restarting playback."""

    def __init__(self, preferences: PreferenceManager, base_path: str = "") -> None:
        self.preferences = preferences
        self.base_path = normalize_url_base_path(base_path)
        self._video_process: subprocess.Popen | None = None
        self._audio_processes: dict[int, subprocess.Popen] = {}
        self._ready: set[int] = set()
        self._rendering: set[int] = set()
        # Pitches counted against MAX_CONCURRENT_BACKGROUND_RENDERS: still
        # alive and still genuinely background (a boost to on-demand drops
        # a pitch out of this - it's no longer discretionary once someone
        # is actually waiting on it).
        self._background_alive: set[int] = set()
        # The pitch currently being listened to. Only this one has to keep
        # pace with playback, so it is the only one held at foreground
        # priority (see _set_active).
        self._active: int | None = None
        self._declared: list[int] = []
        self._pending: list[int] = []
        self._fr: "FileResolver | None" = None
        self._lock = Lock()
        self._stop_event = Event()
        self._bg_thread: Thread | None = None

    def _with_base_path(self, path: str) -> str:
        return f"{self.base_path}{path}" if self.base_path else path

    def is_rendered(self, semitones: int) -> bool:
        with self._lock:
            return semitones in self._ready

    def is_switchable(self, semitones: int) -> bool:
        """Whether this pitch is declared in the current song's master playlist.

        Every supported pitch is advertised to the player up front, whether
        or not it has been rendered yet, so the browser can switch to any
        of them. Only a pitch outside the declared range would need a new
        master playlist, and therefore a restart.
        """
        with self._lock:
            return semitones in self._declared

    def prioritize(self, semitones: int) -> None:
        """Move a queued pitch to the front so it renders next."""
        with self._lock:
            if semitones in self._pending:
                self._pending.remove(semitones)
                self._pending.insert(0, semitones)

    def ensure_switchable(
        self,
        semitones: int,
        position: float = 0,
        timeout: float = SWITCH_WAIT_TIMEOUT_SECONDS,
    ) -> bool:
        """Get a windowed pitch playable enough to switch to at the playhead.

        The rendition has to cover where playback actually is, not just its
        opening: switching to one that has only rendered the first few
        seconds while the song is a minute in leaves the player waiting on
        a segment that doesn't exist yet, which stalls the video too.

        Doesn't need the rendition finished - its playlist grows as ffmpeg
        writes it, so anything past the playhead is enough to switch on.
        """
        fr = self._fr
        if fr is None or not self.is_switchable(semitones):
            return False

        # Pre-rendering only covers a window around the starting key, so a
        # pitch outside it has nothing running yet and nothing queued to
        # wait for - start it here rather than leave the caller waiting on
        # a render that would never happen.
        self.prioritize(semitones)
        started = time.monotonic()
        self._launch_render(fr, semitones, background=False)
        marker = f"{fr.stream_uid}_audio_{semitone_label(semitones)}_"
        playlist = audio_playlist_path(fr, semitones)
        needed = int((position + SWITCH_LOOKAHEAD_SECONDS) // HLS_SEGMENT_SECONDS) + 1

        deadline = time.monotonic() + timeout
        while True:
            if self._stop_event.is_set():
                return False
            if os.path.exists(playlist):
                # A finished rendition counts even if it has fewer segments
                # than the playhead implies - near the end of a song there
                # simply aren't any more to wait for.
                if _count_segments(fr.tmp_dir, marker) >= needed or self._is_complete(semitones):
                    logging.info(
                        f"Pitch {semitones} switchable after " f"{time.monotonic() - started:.1f}s"
                    )
                    self._set_active(semitones)
                    return True
            if time.monotonic() >= deadline:
                logging.info(
                    f"Pitch {semitones} not ready after {timeout:.0f}s, declining the switch"
                )
                return False
            time.sleep(READY_POLL_INTERVAL_SECONDS)

    def _set_active(self, semitones: int) -> None:
        """Make one pitch the foreground render and push every other one down.

        Only the pitch being listened to has to keep pace with playback; the
        rest are discretionary. Without this sweep, stepping through keys
        leaves every pitch touched on the way at foreground priority,
        competing with the one actually playing - six quick steps means six
        rivals, which is enough to starve a later on-demand render.
        """
        with self._lock:
            self._active = semitones
            processes = dict(self._audio_processes)
        for pitch, proc in processes.items():
            if proc.poll() is not None:
                continue
            if pitch == semitones:
                restore_priority(proc)
            else:
                lower_priority(proc)

    def _is_complete(self, semitones: int) -> bool:
        """Whether this rendition's ffmpeg finished writing successfully."""
        with self._lock:
            proc = self._audio_processes.get(semitones)
        return proc is not None and proc.poll() == 0

    def start(self, fr: "FileResolver", base_semitones: int) -> PlaybackResult:
        """Start rendering video plus the base pitch, then render the rest
        of the window per BLOCK_PLAYBACK_UNTIL_WINDOW_RENDERED - either
        before returning, or in the background while the song plays.

        Always blocks until video and the base pitch are ready to play -
        the same latency shape as the legacy single-stream path.
        """
        from flask_babel import _

        self.kill_all()
        self._stop_event.clear()

        window = int(self.preferences.get_or_default("pitch_window_semitones"))
        cdg_pixel_scaling = self.preferences.get_or_default("cdg_pixel_scaling")
        # Declared covers every supported pitch so none of them can force a
        # restart. The window only decides which ones get rendered ahead of
        # being asked for; the rest render on demand when picked.
        declared = sorted({*range(MIN_SEMITONES, MAX_SEMITONES + 1), base_semitones})
        prerendered = sorted(
            s
            for s in range(base_semitones - window, base_semitones + window + 1)
            if MIN_SEMITONES <= s <= MAX_SEMITONES
        )
        with self._lock:
            self._fr = fr
            self._declared = declared

        logging.info(
            f"Pitch pre-render: base={base_semitones}, window=±{window}, " f"declared={declared}"
        )

        video_cmd = build_video_only_ffmpeg_cmd(
            fr,
            video_playlist_path(fr),
            f"{fr.tmp_dir}/{fr.stream_uid}_video_segment_%03d.m4s",
            f"{fr.stream_uid}_video_init.mp4",
            cdg_pixel_scaling,
        )
        self._video_process = video_cmd.run_async(pipe_stderr=True, pipe_stdin=True)
        video_ready = _wait_until_ready(
            fr.tmp_dir,
            f"{fr.stream_uid}_video_",
            video_playlist_path(fr),
            self._video_process,
            self._stop_event,
        )
        if not video_ready:
            self.kill_all()
            return PlaybackResult(success=False, error=_("Failed to prepare video stream"))

        if not self._render_one(fr, base_semitones, background=False):
            self.kill_all()
            return PlaybackResult(success=False, error=_("Failed to prepare audio stream"))
        self._set_active(base_semitones)

        master_path = f"{fr.tmp_dir}/{fr.stream_uid}.m3u8"
        with open(master_path, "w") as f:
            f.write(build_master_playlist(fr, declared, base_semitones))

        # Nearest-to-base first: singers nudge the key a step at a time
        # rather than jumping to the edge of the range, so the pitches most
        # likely to be picked next are ready soonest. A pitch the singer
        # actually asks for jumps this queue (see prioritize).
        with self._lock:
            self._pending = sorted(
                (s for s in prerendered if s != base_semitones),
                key=lambda s: abs(s - base_semitones),
            )
        logging.info(f"Pitch pre-render: queued in background: {self._pending}")
        if BLOCK_PLAYBACK_UNTIL_WINDOW_RENDERED:
            queued_count = len(self._pending)
            window_started = time.monotonic()
            self._render_remaining(fr)
            logging.info(
                f"Pitch pre-render: full window ({queued_count + 1} pitches incl. base) "
                f"rendered in {time.monotonic() - window_started:.1f}s, song starting now"
            )
        else:
            self._bg_thread = Thread(target=self._render_remaining, args=(fr,), daemon=True)
            self._bg_thread.start()

        subtitle_url = None
        if fr.ass_file_path:
            subtitle_url = self._with_base_path(f"/subtitle/{fr.stream_uid}")

        return PlaybackResult(
            success=True,
            stream_url=self._with_base_path(f"/stream/{fr.stream_uid}.m3u8"),
            subtitle_url=subtitle_url,
            duration=fr.duration,
        )

    def _launch_render(
        self, fr: "FileResolver", semitones: int, background: bool = True
    ) -> subprocess.Popen | None:
        """Start one rendition's ffmpeg, unless it is already running.

        Safe to call from both the background queue and an on-demand
        switch, which can race for the same pitch. `background` sets this
        process's OS scheduling priority every time this is called for it,
        whether it was just started or was already running - so a render
        the background queue started gets bumped back to normal priority
        the moment someone actually asks for that pitch, rather than
        staying deprioritized while they wait on it.
        """
        with self._lock:
            existing = self._audio_processes.get(semitones)
            already_running = semitones in self._rendering and existing is not None
            if not already_running:
                self._rendering.add(semitones)
                if semitones in self._pending:
                    self._pending.remove(semitones)

        if already_running:
            proc = existing
            if not background:
                with self._lock:
                    self._background_alive.discard(semitones)
                logging.info(f"Pitch {semitones} already rendering in background, prioritizing it")
        else:
            normalize_audio = self.preferences.get_or_default("normalize_audio")
            avsync = self.preferences.get_or_default("avsync")
            label = semitone_label(semitones)
            cmd = build_audio_only_ffmpeg_cmd(
                fr,
                semitones,
                audio_playlist_path(fr, semitones),
                f"{fr.tmp_dir}/{fr.stream_uid}_audio_{label}_segment_%03d.m4s",
                f"{fr.stream_uid}_audio_{label}_init.mp4",
                normalize_audio,
                avsync,
            )
            proc = cmd.run_async(pipe_stderr=True, pipe_stdin=True)
            started = time.monotonic()
            with self._lock:
                self._audio_processes[semitones] = proc
                if background:
                    self._background_alive.add(semitones)
            logging.info(
                f"Rendering pitch {semitones} " f"({'background' if background else 'on-demand'})"
            )
            Thread(
                target=self._log_full_completion, args=(semitones, proc, started), daemon=True
            ).start()

        if background:
            lower_priority(proc)
        else:
            restore_priority(proc)
        return proc

    def _log_full_completion(self, semitones: int, proc: subprocess.Popen, started: float) -> None:
        """Log when a rendition's ffmpeg actually finishes writing the whole
        song - separate from "ready" above, which only needs the first few
        segments. Ready times alone can't tell a slow queue from a slow
        render; this is the number that answers that.
        """
        exit_code = proc.wait()
        if self._stop_event.is_set():
            # A nonzero exit here is us tearing this process down on
            # purpose (e.g. a restart superseded it), not worth logging.
            return
        elapsed = time.monotonic() - started
        if exit_code == 0:
            logging.info(f"Pitch {semitones} finished rendering (full song) in {elapsed:.1f}s")
        else:
            logging.debug(
                f"Pitch {semitones} render process exited with code {exit_code} "
                f"after {elapsed:.1f}s"
            )

    def _render_one(self, fr: "FileResolver", semitones: int, background: bool = True) -> bool:
        """Render one audio-only rendition and mark it ready to play from."""
        label = semitone_label(semitones)
        started = time.monotonic()
        proc = self._launch_render(fr, semitones, background)
        if proc is None:
            return False
        ready = _wait_until_ready(
            fr.tmp_dir,
            f"{fr.stream_uid}_audio_{label}_",
            audio_playlist_path(fr, semitones),
            proc,
            self._stop_event,
        )
        if ready:
            with self._lock:
                self._ready.add(semitones)
            logging.info(f"Pitch {semitones} ready after {time.monotonic() - started:.1f}s")
        return ready

    def _background_slot_available(self) -> bool:
        """Prune background renders that finished or got boosted, then
        report whether there's room for another under the concurrency cap."""
        with self._lock:
            self._background_alive = {
                s
                for s in self._background_alive
                if (proc := self._audio_processes.get(s)) is not None and proc.poll() is None
            }
            return len(self._background_alive) < MAX_CONCURRENT_BACKGROUND_RENDERS

    def _render_remaining(self, fr: "FileResolver") -> None:
        """Render the rest of the pitch window in the background.

        Each queued pitch is launched once a background slot is free, not
        once the previous one has fully finished encoding, which can take
        the length of the whole song. A rendition that's past ready keeps
        writing in the background at low priority behind the scenes; the
        cap exists because letting those pile up unbounded made each
        successive render slower than the last, as they increasingly
        starved each other for CPU. Renditions are pulled from a queue
        rather than a fixed list so a pitch the singer asks for can jump
        ahead of the ones queued near it.
        """
        while not self._stop_event.is_set():
            with self._lock:
                pending_empty = not self._pending
            if pending_empty:
                logging.info("Pitch pre-render queue empty, background rendering done")
                return
            if not self._background_slot_available():
                # Nothing left to render is checked above, before this -
                # otherwise an empty queue with slots still occupied by
                # long-running renders would wait here forever for
                # capacity it will never need.
                time.sleep(READY_POLL_INTERVAL_SECONDS)
                continue
            with self._lock:
                if not self._pending:
                    logging.info("Pitch pre-render queue empty, background rendering done")
                    return
                semitones = self._pending.pop(0)
            self._render_one(fr, semitones)

    def kill_all(self) -> None:
        """Tear down every process this manager owns (video + all audio renditions)."""
        self._stop_event.set()
        with self._lock:
            audio_processes = list(self._audio_processes.values())
            self._audio_processes.clear()
            self._ready.clear()
            self._rendering.clear()
            self._background_alive.clear()
            self._active = None
            self._declared = []
            self._pending = []
            self._fr = None
            video_process = self._video_process
            self._video_process = None
        _terminate(video_process)
        for proc in audio_processes:
            _terminate(proc)
        if self._bg_thread and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=1)
