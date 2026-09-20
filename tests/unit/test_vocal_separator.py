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
    def test_returns_the_cached_track_without_running_demucs(self, separator, songs_dir):
        song = make_song(songs_dir)
        track = write_cache(separator, song)
        with patch("pikaraoke.lib.vocal_separator.subprocess.Popen") as popen:
            assert separator.separate(song) == track
        popen.assert_not_called()

    def test_gives_up_when_the_submodule_is_missing(self, separator, songs_dir):
        song = make_song(songs_dir)
        with patch("pikaraoke.lib.vocal_separator._vendored_demucs_dir", return_value=None):
            with patch("pikaraoke.lib.vocal_separator.subprocess.Popen") as popen:
                assert separator.separate(song) is None
        popen.assert_not_called()

    def test_caches_what_demucs_produced(self, separator, songs_dir, events):
        song = make_song(songs_dir)
        separated = []
        events.on("vocals_separated", lambda path, track: separated.append((path, track)))

        def fake_demucs(cmd, **kwargs):
            # Write into the -o directory the way the demucs CLI does, under a
            # subfolder named for the model.
            out_dir = cmd[cmd.index("-o") + 1]
            track_dir = os.path.join(out_dir, "htdemucs", "song")
            os.makedirs(track_dir)
            with open(os.path.join(track_dir, "no_vocals.wav"), "wb") as f:
                f.write(b"lossless-backing-track")
            process = MagicMock()
            process.communicate.return_value = ("", None)
            process.returncode = 0
            return process

        def fake_encode(wav_path, mp3_path):
            with open(wav_path, "rb") as f:
                assert f.read() == b"lossless-backing-track"
            with open(mp3_path, "wb") as f:
                f.write(b"backing-track")
            return True

        with patch("pikaraoke.lib.vocal_separator._vendored_demucs_dir", return_value="/demucs"):
            with patch("pikaraoke.lib.vocal_separator.subprocess.Popen", side_effect=fake_demucs):
                with patch("pikaraoke.lib.vocal_separator.use_spare_capacity"):
                    with patch("pikaraoke.lib.vocal_separator._encode_mp3", fake_encode):
                        track = separator.separate(song)

        assert track is not None
        with open(track, "rb") as f:
            assert f.read() == b"backing-track"
        # Cached, so a second call is free.
        assert separator.cached_track(song) == track
        assert separated == [(song, track)]

    def test_nothing_is_cached_when_demucs_fails(self, separator, songs_dir):
        song = make_song(songs_dir)
        process = MagicMock()
        process.communicate.return_value = ("boom", None)
        process.returncode = 1

        with patch("pikaraoke.lib.vocal_separator._vendored_demucs_dir", return_value="/demucs"):
            with patch("pikaraoke.lib.vocal_separator.subprocess.Popen", return_value=process):
                with patch("pikaraoke.lib.vocal_separator.use_spare_capacity"):
                    assert separator.separate(song) is None

        assert separator.cached_track(song) is None

    def test_runs_the_vendored_fork_not_an_installed_demucs(self, separator, songs_dir):
        """The fork is the point of vendoring it, so it must win the import."""
        song = make_song(songs_dir)
        process = MagicMock()
        process.communicate.return_value = ("", None)
        process.returncode = 1
        captured = {}

        def capture(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs["env"]
            return process

        with patch(
            "pikaraoke.lib.vocal_separator._vendored_demucs_dir", return_value="/vendor/demucs"
        ):
            with patch("pikaraoke.lib.vocal_separator.subprocess.Popen", side_effect=capture):
                with patch("pikaraoke.lib.vocal_separator.use_spare_capacity"):
                    separator.separate(song)

        assert captured["env"]["PYTHONPATH"].split(os.pathsep)[0] == "/vendor/demucs"
        assert "--two-stems" in captured["cmd"]
        assert captured["cmd"][captured["cmd"].index("--two-stems") + 1] == "vocals"
        # The mp3 is made by ffmpeg afterwards, so demucs must not make its own.
        assert "--mp3" not in captured["cmd"]


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
        """Demucs' own mp3 encoder left the track 1105 samples (25ms) long and late,
        which is an audible jump when toggling vocals against the original audio."""
        wav = str(tmp_path / "in.wav")
        mp3 = str(tmp_path / "out.mp3")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=d=3:r=44100", "-ac", "2", wav],
            check=True,
        )
        assert _encode_mp3(wav, mp3)

        def sample_count(path):
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "2", "-"],
                capture_output=True,
                check=True,
            ).stdout
            return len(raw) // 4

        assert abs(sample_count(mp3) - sample_count(wav)) < 10

    def test_reports_failure_for_an_unreadable_input(self, tmp_path):
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not audio")
        assert not _encode_mp3(str(bad), str(tmp_path / "out.mp3"))
