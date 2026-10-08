"""Unit tests for vocal_separator module."""

import json
import os
import shutil
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.ffmpeg import PCM_BYTES_PER_SECOND
from pikaraoke.lib.preference_manager import PreferenceManager
from pikaraoke.lib.vocal_separator import (
    BACKGROUND,
    BEFORE_PLAY,
    OFF,
    VocalSeparator,
    _encode_mp3,
    stems_dir,
)


@pytest.fixture
def events():
    return EventSystem()


@pytest.fixture
def preferences(tmp_path):
    return PreferenceManager(str(tmp_path / "config.ini"))


@pytest.fixture
def songs_dir(tmp_path):
    d = tmp_path / "songs"
    d.mkdir()
    return str(d)


@pytest.fixture
def separator(events, preferences, songs_dir):
    return VocalSeparator(events=events, preferences=preferences, songs_dir=songs_dir)


def make_song(songs_dir: str, name: str = "Artist - Song---dQw4w9WgXcQ.mp4") -> str:
    path = os.path.join(songs_dir, name)
    with open(path, "wb") as f:
        f.write(b"video-bytes")
    return path


def write_cache(separator: VocalSeparator, song_path: str, fingerprint: dict | None = None) -> str:
    """Place a cached track and its fingerprint as a real separation would."""
    stat = os.stat(song_path)
    directory = stems_dir(separator._songs_dir)
    os.makedirs(directory, exist_ok=True)
    key = "dQw4w9WgXcQ"
    track = os.path.join(directory, key + ".no_vocals.mp3")
    with open(track, "wb") as f:
        f.write(b"audio-bytes")
    with open(os.path.join(directory, key + ".json"), "w", encoding="utf-8") as f:
        json.dump(
            fingerprint or {"size": stat.st_size, "mtime": int(stat.st_mtime)},
            f,
        )
    return track


class TestMode:
    def test_defaults_to_off(self, separator):
        assert separator.mode == OFF

    def test_reads_the_preference(self, separator, preferences):
        preferences.set("vocal_separation", BACKGROUND)
        assert separator.mode == BACKGROUND

    def test_unknown_mode_falls_back_to_off(self, separator, preferences):
        """A hand-edited config must not turn into an unhandled mode."""
        preferences.set("vocal_separation", "sideways")
        assert separator.mode == OFF


class TestCachedTrack:
    def test_miss_when_nothing_separated(self, separator, songs_dir):
        assert separator.cached_track(make_song(songs_dir)) is None

    def test_hit_when_fingerprint_matches(self, separator, songs_dir):
        song = make_song(songs_dir)
        track = write_cache(separator, song)
        assert separator.cached_track(song) == track

    def test_miss_when_source_changed(self, separator, songs_dir):
        """A re-downloaded song must not play the previous file's backing track."""
        song = make_song(songs_dir)
        write_cache(separator, song, fingerprint={"size": 999, "mtime": 1})
        assert separator.cached_track(song) is None

    def test_miss_when_fingerprint_unreadable(self, separator, songs_dir):
        song = make_song(songs_dir)
        write_cache(separator, song)
        with open(os.path.join(stems_dir(separator._songs_dir), "dQw4w9WgXcQ.json"), "w") as f:
            f.write("not json")
        assert separator.cached_track(song) is None

    def test_cache_survives_a_rename_of_a_youtube_song(self, separator, songs_dir):
        """The cache is keyed on the YouTube ID, which a rename preserves."""
        song = make_song(songs_dir)
        write_cache(separator, song)
        renamed = os.path.join(songs_dir, "Better Name---dQw4w9WgXcQ.mp4")
        os.rename(song, renamed)
        assert separator.cached_track(renamed) is not None


class TestRemoveCached:
    def test_removes_track_and_fingerprint(self, separator, songs_dir):
        song = make_song(songs_dir)
        write_cache(separator, song)
        separator.remove_cached(song)
        assert separator.cached_track(song) is None
        assert os.listdir(stems_dir(separator._songs_dir)) == []

    def test_is_quiet_when_nothing_is_cached(self, separator, songs_dir):
        separator.remove_cached(make_song(songs_dir))


def partial_path(separator: VocalSeparator) -> str:
    return os.path.join(stems_dir(separator._songs_dir), "dQw4w9WgXcQ.partial.pcm")


class TestQueueSeparation:
    def test_skips_an_already_cached_song(self, separator, songs_dir):
        song = make_song(songs_dir)
        write_cache(separator, song)
        separator.queue_separation(song)
        assert separator._pending == []

    def test_queues_an_unseparated_song(self, separator, songs_dir):
        song = make_song(songs_dir)
        separator.queue_separation(song)
        assert separator._pending == [song]

    def test_a_song_is_only_in_line_once(self, separator, songs_dir):
        song = make_song(songs_dir)
        separator.queue_separation(song)
        separator.queue_separation(song)
        separator._current = make_song(songs_dir, "Other---aaaaaaaaaaa.mp4")
        separator.queue_separation(separator._current)
        assert separator._pending == [song]


class TestPrioritize:
    def test_moves_the_song_to_the_front(self, separator, songs_dir):
        first, second = make_song(songs_dir, "A---aaaaaaaaaaa.mp4"), make_song(songs_dir)
        separator.queue_separation(first)
        separator.queue_separation(second)

        separator.prioritize(second)

        assert separator._pending == [second, first]

    def test_interrupts_another_song_and_puts_it_back_right_behind(self, separator, songs_dir):
        """The song about to play can't wait a minute for another one to finish first."""
        playing, other = make_song(songs_dir), make_song(songs_dir, "B---bbbbbbbbbbb.mp4")
        later = make_song(songs_dir, "C---ccccccccccc.mp4")
        separator._current = other
        separator._process = MagicMock()
        separator.queue_separation(later)

        separator.prioritize(playing)

        separator._process.terminate.assert_called_once()
        assert separator._pending == [playing, other, later]

    def test_leaves_the_song_already_separating_alone(self, separator, songs_dir):
        song = make_song(songs_dir)
        separator._current = song
        separator._process = MagicMock()

        separator.prioritize(song)

        separator._process.terminate.assert_not_called()
        assert separator._pending == []

    def test_does_nothing_for_a_cached_song(self, separator, songs_dir):
        song = make_song(songs_dir)
        write_cache(separator, song)
        separator.prioritize(song)
        assert separator._pending == []


class TestProgress:
    def test_partial_track_only_for_the_song_separating(self, separator, songs_dir):
        song = make_song(songs_dir)
        assert separator.partial_track(song) is None
        separator._current = song
        assert separator.partial_track(song) == partial_path(separator)

    def test_seconds_ready_counts_what_has_been_written(self, separator, songs_dir):
        song = make_song(songs_dir)
        separator._current = song
        os.makedirs(stems_dir(separator._songs_dir))
        with open(partial_path(separator), "wb") as f:
            f.write(b"\0" * PCM_BYTES_PER_SECOND * 3)
        assert separator.seconds_ready(song) == 3

    def test_a_cached_song_is_ready_all_the_way(self, separator, songs_dir):
        song = make_song(songs_dir)
        write_cache(separator, song)
        assert separator.seconds_ready(song) == float("inf")

    def test_a_song_not_being_separated_has_nothing_ready(self, separator, songs_dir):
        assert separator.seconds_ready(make_song(songs_dir)) == 0


class TestWaitForHeadStart:
    def test_returns_once_enough_is_separated(self, separator, songs_dir):
        song = make_song(songs_dir)
        separator._current = song
        with patch.object(separator, "seconds_ready", side_effect=[2, 9, 16]):
            with patch("pikaraoke.lib.vocal_separator.sleep"):
                assert separator.wait_for_head_start(song, seconds=15) is True

    def test_gives_up_when_separation_ends_without_a_track(self, separator, songs_dir):
        """It isn't separating, isn't queued and isn't cached: it failed, so don't hang."""
        song = make_song(songs_dir)
        with patch.object(separator, "prioritize"):
            with patch("pikaraoke.lib.vocal_separator.sleep") as slept:
                assert separator.wait_for_head_start(song) is False
        slept.assert_not_called()

    def test_puts_the_song_first_in_line(self, separator, songs_dir):
        song = make_song(songs_dir)
        write_cache(separator, song)
        with patch.object(separator, "prioritize") as prioritized:
            separator.wait_for_head_start(song)
        prioritized.assert_called_once_with(song)


class TestSeparate:
    @staticmethod
    def fake_separator(pcm: bytes = b"raw-backing-track", returncode: int = 0):
        """A Popen stand-in that writes `pcm` where the separator writes its output."""
        calls = []

        def popen(cmd, **kwargs):
            calls.append(cmd)
            if returncode == 0:
                with open(cmd[4], "wb") as f:
                    f.write(pcm)
            process = MagicMock()
            process.communicate.return_value = ("" if returncode == 0 else "boom", None)
            process.returncode = returncode
            return process

        return popen, calls

    @staticmethod
    def run_next(separator, song, popen):
        separator._pending = [song]
        with patch("pikaraoke.lib.vocal_separator.subprocess.Popen", side_effect=popen):
            with patch("pikaraoke.lib.vocal_separator.use_spare_capacity"):
                separator._separate_next()

    def test_skips_a_song_cached_since_it_was_queued(self, separator, songs_dir):
        song = make_song(songs_dir)
        write_cache(separator, song)
        popen, calls = self.fake_separator()
        self.run_next(separator, song, popen)
        assert calls == []

    def test_caches_what_the_separator_produced(self, separator, songs_dir, events):
        song = make_song(songs_dir)
        separated = []
        events.on("vocals_separated", lambda path, track: separated.append((path, track)))
        popen, _calls = self.fake_separator()

        def fake_encode(pcm_path, mp3_path):
            with open(pcm_path, "rb") as f:
                assert f.read() == b"raw-backing-track"
            with open(mp3_path, "wb") as f:
                f.write(b"backing-track")
            return True

        with patch("pikaraoke.lib.vocal_separator._encode_mp3", fake_encode):
            self.run_next(separator, song, popen)

        track = separator.cached_track(song)
        with open(track, "rb") as f:
            assert f.read() == b"backing-track"
        assert separated == [(song, track)]

    def test_writes_the_partial_track_where_readers_follow_it(self, separator, songs_dir):
        song = make_song(songs_dir)
        popen, calls = self.fake_separator(returncode=1)
        self.run_next(separator, song, popen)
        assert calls[0][4] == partial_path(separator)

    def test_announces_the_start_so_the_vocals_button_can_switch_on(
        self, separator, songs_dir, events
    ):
        song = make_song(songs_dir)
        started = []
        events.on("vocals_separating", started.append)
        popen, _calls = self.fake_separator(returncode=1)
        self.run_next(separator, song, popen)
        assert started == [song]

    def test_nothing_is_cached_when_separation_fails(self, separator, songs_dir):
        song = make_song(songs_dir)
        popen, _calls = self.fake_separator(returncode=1)
        self.run_next(separator, song, popen)
        assert separator.cached_track(song) is None

    def test_an_interrupted_separation_is_not_cached(self, separator, songs_dir):
        """It was cut short to make way for the song about to play, and starts over later."""
        song = make_song(songs_dir)
        popen, _calls = self.fake_separator(returncode=-15)

        def interrupted(cmd, **kwargs):
            separator._interrupted = True
            return popen(cmd, **kwargs)

        with patch("pikaraoke.lib.vocal_separator._encode_mp3") as encode:
            self.run_next(separator, song, interrupted)
        encode.assert_not_called()

    def test_runs_the_separator_in_its_own_process(self, separator, songs_dir):
        """In-process, its number crunching would stall the web server between chunks."""
        song = make_song(songs_dir)
        popen, calls = self.fake_separator(returncode=1)
        separator._pending = [song]
        with patch("pikaraoke.lib.vocal_separator.subprocess.Popen", side_effect=popen):
            with patch("pikaraoke.lib.vocal_separator.use_spare_capacity") as lowered:
                with patch(
                    "pikaraoke.lib.vocal_separator.get_data_directory", return_value="/data"
                ):
                    separator._separate_next()

        cmd = calls[0]
        assert cmd[1:4] == ["-m", "pikaraoke.lib.mdx", song]
        assert cmd[cmd.index("--model-dir") + 1] == os.path.join("/data", "models")
        lowered.assert_called_once()

    def test_the_worker_is_free_again_afterwards(self, separator, songs_dir):
        song = make_song(songs_dir)
        popen, _calls = self.fake_separator(returncode=1)
        self.run_next(separator, song, popen)
        assert separator._current is None
        assert separator.partial_track(song) is None


class TestSweepPartials:
    def test_deletes_partial_tracks_no_longer_being_written(self, separator, songs_dir):
        separating = make_song(songs_dir)
        separator._current = separating
        directory = stems_dir(separator._songs_dir)
        os.makedirs(directory)
        for name in (
            "dQw4w9WgXcQ.partial.pcm",
            "aaaaaaaaaaa.partial.pcm",
            "bbbbbbbbbbb.no_vocals.mp3",
        ):
            with open(os.path.join(directory, name), "wb") as f:
                f.write(b"x")

        separator._sweep_partials()

        assert sorted(os.listdir(directory)) == [
            "bbbbbbbbbbb.no_vocals.mp3",
            "dQw4w9WgXcQ.partial.pcm",
        ]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs a real ffmpeg")
class TestEncodeMp3:
    def test_encoded_track_decodes_to_the_same_length_as_the_source(self, tmp_path):
        """An mp3 encoder that doesn't record its 1105-sample delay leaves the track
        25ms late, which is an audible jump when toggling vocals against the original."""
        pcm = str(tmp_path / "in.pcm")
        mp3 = str(tmp_path / "out.mp3")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=d=3:r=44100"]
            + ["-ac", "2", "-f", "s16le", pcm],
            check=True,
        )
        assert _encode_mp3(pcm, mp3)

        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", mp3, "-f", "s16le", "-ac", "2", "-"],
            capture_output=True,
            check=True,
        ).stdout
        assert abs(len(decoded) // 4 - os.path.getsize(pcm) // 4) < 10

    def test_reports_failure_for_a_missing_input(self, tmp_path):
        assert not _encode_mp3(str(tmp_path / "missing.pcm"), str(tmp_path / "out.mp3"))
