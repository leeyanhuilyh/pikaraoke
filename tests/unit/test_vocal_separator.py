"""Unit tests for vocal_separator module."""

import json
import os
import shutil
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pikaraoke.lib.events import EventSystem
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


class TestQueueSeparation:
    def test_skips_an_already_cached_song(self, separator, songs_dir):
        song = make_song(songs_dir)
        write_cache(separator, song)
        separator.queue_separation(song)
        assert separator._queue.empty()

    def test_queues_an_unseparated_song(self, separator, songs_dir):
        separator.queue_separation(make_song(songs_dir))
        assert separator._queue.qsize() == 1


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

    def test_returns_the_cached_track_without_running_the_separator(self, separator, songs_dir):
        song = make_song(songs_dir)
        track = write_cache(separator, song)
        with patch("pikaraoke.lib.vocal_separator.subprocess.Popen") as popen:
            assert separator.separate(song) == track
        popen.assert_not_called()

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

        with patch("pikaraoke.lib.vocal_separator.subprocess.Popen", side_effect=popen):
            with patch("pikaraoke.lib.vocal_separator.use_spare_capacity"):
                with patch("pikaraoke.lib.vocal_separator._encode_mp3", fake_encode):
                    track = separator.separate(song)

        assert track is not None
        with open(track, "rb") as f:
            assert f.read() == b"backing-track"
        # Cached, so a second call is free.
        assert separator.cached_track(song) == track
        assert separated == [(song, track)]

    def test_nothing_is_cached_when_separation_fails(self, separator, songs_dir):
        song = make_song(songs_dir)
        popen, _calls = self.fake_separator(returncode=1)

        with patch("pikaraoke.lib.vocal_separator.subprocess.Popen", side_effect=popen):
            with patch("pikaraoke.lib.vocal_separator.use_spare_capacity"):
                assert separator.separate(song) is None

        assert separator.cached_track(song) is None

    def test_runs_the_separator_in_its_own_process(self, separator, songs_dir):
        """In-process, its number crunching would stall the web server between chunks."""
        song = make_song(songs_dir)
        popen, calls = self.fake_separator(returncode=1)

        with patch("pikaraoke.lib.vocal_separator.subprocess.Popen", side_effect=popen):
            with patch("pikaraoke.lib.vocal_separator.use_spare_capacity") as lowered:
                with patch(
                    "pikaraoke.lib.vocal_separator.get_data_directory", return_value="/data"
                ):
                    separator.separate(song)

        cmd = calls[0]
        assert cmd[1:4] == ["-m", "pikaraoke.lib.mdx", song]
        assert cmd[cmd.index("--model-dir") + 1] == os.path.join("/data", "models")
        lowered.assert_called_once()


class TestConcurrentSeparation:
    def test_a_second_caller_reuses_the_first_ones_result(self, separator, songs_dir):
        """The queue worker can be mid-song when playback reaches it. Running demucs
        twice for one song would double the most expensive operation in the app."""
        song = make_song(songs_dir)
        runs = []

        def run_once(path):
            runs.append(path)
            return write_cache(separator, path)

        with patch.object(separator, "_separate_uncached", side_effect=run_once):
            first = separator.separate(song)
            second = separator.separate(song)

        assert first == second
        assert runs == [song]


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
