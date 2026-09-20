"""Unit tests for rendition_manager module."""

from unittest.mock import MagicMock, patch

import pytest

from pikaraoke.lib.preference_manager import PreferenceManager
from pikaraoke.lib.rendition_manager import (
    MAX_SEMITONES,
    MIN_SEMITONES,
    Rendition,
    RenditionManager,
    audio_playlist_path,
    build_master_playlist,
    rendition_label,
    rendition_name,
    semitone_label,
    video_playlist_path,
)


@pytest.fixture
def test_prefs():
    """Create a PreferenceManager for testing."""
    return PreferenceManager("/nonexistent/test_config.ini")


def _separator(mode="background", track="/songs/.stems/x.no_vocals.mp3"):
    """A vocal separator stand-in in a given mode, with or without a finished track."""
    sep = MagicMock()
    sep.mode = mode
    sep.cached_track.return_value = track
    return sep


def _range_of(semitones, vocals_on=True):
    return [Rendition(s, vocals_on) for s in semitones]


def _make_mock_fr(tmp_dir="/tmp", stream_uid=12345, ass_file_path=None, duration=180):
    mock_fr = MagicMock()
    mock_fr.tmp_dir = tmp_dir
    mock_fr.stream_uid = stream_uid
    mock_fr.ass_file_path = ass_file_path
    mock_fr.duration = duration
    return mock_fr


class TestSemitoneLabel:
    """Tests for the semitone_label filename token."""

    def test_positive(self):
        assert semitone_label(4) == "p4"

    def test_negative(self):
        assert semitone_label(-4) == "m4"

    def test_zero(self):
        assert semitone_label(0) == "p0"


class TestPaths:
    """Tests for the rendition path helpers."""

    def test_video_playlist_path(self):
        fr = _make_mock_fr()
        assert video_playlist_path(fr) == "/tmp/12345_video.m3u8"

    def test_audio_playlist_path(self):
        fr = _make_mock_fr()
        assert audio_playlist_path(fr, Rendition(-3, True)) == "/tmp/12345_audio_m3.m3u8"

    def test_audio_playlist_path_without_vocals(self):
        fr = _make_mock_fr()
        assert audio_playlist_path(fr, Rendition(-3, False)) == "/tmp/12345_audio_m3nv.m3u8"


class TestRenditionNaming:
    """The label keys files and markers, the name is what the splash player parses."""

    def test_label_marks_only_the_vocals_off_rendition(self):
        assert rendition_label(Rendition(4, True)) == "p4"
        assert rendition_label(Rendition(4, False)) == "p4nv"

    def test_no_vocals_label_never_collides_with_another_pitch_marker(self):
        """Segment counts match on "<label>_", so p1nv must not look like p1 or p12."""
        markers = {
            rendition_label(Rendition(s, v)) + "_"
            for s in range(MIN_SEMITONES, MAX_SEMITONES + 1)
            for v in (True, False)
        }
        assert len(markers) == 2 * (MAX_SEMITONES - MIN_SEMITONES + 1)
        assert not any(a != b and b.startswith(a) for a in markers for b in markers)

    def test_name_carries_pitch_and_vocals_state(self):
        assert rendition_name(Rendition(3, True)) == "3:on"
        assert rendition_name(Rendition(-2, False)) == "-2:off"


class TestBuildMasterPlaylist:
    """Tests for build_master_playlist."""

    def test_declares_one_media_entry_per_offset(self):
        fr = _make_mock_fr()
        playlist = build_master_playlist(fr, _range_of([-1, 0, 1]), Rendition(0))

        assert playlist.count("#EXT-X-MEDIA:TYPE=AUDIO") == 3
        assert playlist.count("#EXT-X-STREAM-INF") == 1
        assert "12345_video.m3u8" in playlist

    def test_default_flag_on_base_semitone_only(self):
        fr = _make_mock_fr()
        renditions = _range_of([-1, 0, 1]) + _range_of([-1, 0, 1], vocals_on=False)
        playlist = build_master_playlist(fr, renditions, Rendition(1))

        default_lines = [line for line in playlist.splitlines() if "DEFAULT=YES" in line]
        assert len(default_lines) == 1
        assert 'NAME="1:on"' in default_lines[0]

    def test_a_player_never_auto_selects_the_vocals_off_track(self):
        fr = _make_mock_fr()
        renditions = _range_of([0]) + _range_of([0], vocals_on=False)
        playlist = build_master_playlist(fr, renditions, Rendition(0))

        off_line = next(line for line in playlist.splitlines() if 'NAME="0:off"' in line)
        assert "AUTOSELECT=NO" in off_line

    def test_every_pitch_gets_a_distinct_uri(self):
        fr = _make_mock_fr()
        renditions = _range_of(range(-6, 7)) + _range_of(range(-6, 7), vocals_on=False)
        playlist = build_master_playlist(fr, renditions, Rendition(0))

        uris = [line.split('URI="')[1] for line in playlist.splitlines() if 'URI="' in line]
        assert len(uris) == len(renditions)
        assert len(set(uris)) == len(renditions)


class TestRenditionManagerStart:
    """Tests for RenditionManager.start."""

    @patch("pikaraoke.lib.rendition_manager._wait_until_ready", return_value=True)
    @patch("pikaraoke.lib.rendition_manager.build_audio_only_ffmpeg_cmd")
    @patch("pikaraoke.lib.rendition_manager.build_video_only_ffmpeg_cmd")
    def test_start_success_returns_master_playlist_url(
        self, mock_video_cmd, mock_audio_cmd, mock_ready, test_prefs, tmp_path
    ):
        test_prefs.set("pitch_window_semitones", 2)
        mock_video_cmd.return_value.run_async.return_value = MagicMock()
        mock_audio_cmd.return_value.run_async.return_value = MagicMock()
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))

        with patch.object(rm, "_render_remaining"):
            result = rm.start(fr, base_semitones=0)

        assert result.success is True
        assert result.stream_url == f"/stream/{fr.stream_uid}.m3u8"
        assert (tmp_path / f"{fr.stream_uid}.m3u8").exists()

    @patch("pikaraoke.lib.rendition_manager._wait_until_ready", return_value=False)
    @patch("pikaraoke.lib.rendition_manager.build_video_only_ffmpeg_cmd")
    def test_start_fails_if_video_never_becomes_ready(
        self, mock_video_cmd, mock_ready, test_prefs, tmp_path
    ):
        mock_video_cmd.return_value.run_async.return_value = MagicMock()
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))

        result = rm.start(fr, base_semitones=0)

        assert result.success is False

    @patch("pikaraoke.lib.rendition_manager._wait_until_ready", return_value=True)
    @patch("pikaraoke.lib.rendition_manager.build_audio_only_ffmpeg_cmd")
    @patch("pikaraoke.lib.rendition_manager.build_video_only_ffmpeg_cmd")
    def test_declares_the_full_range_regardless_of_the_window(
        self, mock_video_cmd, mock_audio_cmd, mock_ready, test_prefs, tmp_path
    ):
        """Every supported pitch is switchable; the window is only a head start."""
        test_prefs.set("pitch_window_semitones", 2)
        mock_video_cmd.return_value.run_async.return_value = MagicMock()
        mock_audio_cmd.return_value.run_async.return_value = MagicMock()
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))

        with patch.object(rm, "_render_remaining"):
            rm.start(fr, base_semitones=0)

        assert rm._declared == _range_of(range(MIN_SEMITONES, MAX_SEMITONES + 1))
        # Nearest-to-base first, so the pitches a singer is most likely to
        # step to next are ready soonest.
        assert rm._pending == _range_of([-1, 1, -2, 2])

    @patch("pikaraoke.lib.rendition_manager._wait_until_ready", return_value=True)
    @patch("pikaraoke.lib.rendition_manager.build_audio_only_ffmpeg_cmd")
    @patch("pikaraoke.lib.rendition_manager.build_video_only_ffmpeg_cmd")
    def test_window_clamped_to_the_supported_range(
        self, mock_video_cmd, mock_audio_cmd, mock_ready, test_prefs, tmp_path
    ):
        test_prefs.set("pitch_window_semitones", 5)
        mock_video_cmd.return_value.run_async.return_value = MagicMock()
        mock_audio_cmd.return_value.run_async.return_value = MagicMock()
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))

        with patch.object(rm, "_render_remaining"):
            rm.start(fr, base_semitones=4)

        assert max(r.semitones for r in rm._pending) == MAX_SEMITONES
        assert min(r.semitones for r in rm._pending) == -1


class TestRenditionManagerSequentialRendering:
    """Tests for _render_remaining's sequential-not-concurrent guarantee."""

    def test_renders_queue_in_order(self, test_prefs, tmp_path):
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._pending = _range_of([1, 2, 3])
        call_order = []

        def fake_render_one(fr_arg, rendition):
            call_order.append(rendition.semitones)
            return True

        with patch.object(rm, "_render_one", side_effect=fake_render_one):
            rm._render_remaining(fr, Rendition(0))

        assert call_order == [1, 2, 3]

    def test_waits_for_each_render_before_starting_the_next(self, test_prefs, tmp_path):
        """Renditions must not pile up as concurrent encodes on a Pi."""
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._pending = _range_of([1, 2])
        events = []

        def fake_render_one(fr_arg, rendition):
            semitones = rendition.semitones
            events.append(f"start-{semitones}")
            proc = MagicMock()
            proc.wait.side_effect = lambda: events.append(f"wait-{semitones}")
            rm._audio_processes[rendition] = proc
            return True

        with patch.object(rm, "_render_one", side_effect=fake_render_one):
            rm._render_remaining(fr, Rendition(0))

        assert events == ["start-1", "wait-1", "start-2", "wait-2"]

    def test_stops_when_stop_event_set_mid_sequence(self, test_prefs, tmp_path):
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._pending = _range_of([1, 2, 3])
        call_order = []

        def fake_render_one(fr_arg, rendition):
            call_order.append(rendition.semitones)
            if rendition.semitones == 1:
                rm._stop_event.set()
            return True

        with patch.object(rm, "_render_one", side_effect=fake_render_one):
            rm._render_remaining(fr, Rendition(0))

        assert call_order == [1]

    def test_prioritize_moves_a_queued_pitch_to_the_front(self, test_prefs, tmp_path):
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._pending = _range_of([1, -1, 2, -2])
        call_order = []

        def fake_render_one(fr_arg, rendition):
            call_order.append(rendition.semitones)
            return True

        rm.prioritize(Rendition(-2))

        with patch.object(rm, "_render_one", side_effect=fake_render_one):
            rm._render_remaining(fr, Rendition(0))

        assert call_order == [-2, 1, -1, 2]

    def test_prioritize_ignores_a_pitch_that_is_not_queued(self, test_prefs):
        rm = RenditionManager(test_prefs)
        rm._pending = _range_of([1, 2])

        rm.prioritize(Rendition(9))

        assert rm._pending == _range_of([1, 2])


class TestRenditionManagerIsSwitchable:
    """Tests for is_switchable, which decides switch vs restart."""

    def test_true_for_a_declared_pitch(self, test_prefs):
        rm = RenditionManager(test_prefs)
        rm._declared = _range_of([-2, -1, 0, 1, 2])

        assert rm.is_switchable(2) is True

    def test_true_even_when_not_rendered_or_pre_rendered(self, test_prefs):
        """The master playlist advertises it, so the player can switch to it."""
        rm = RenditionManager(test_prefs)
        rm._declared = _range_of([-2, -1, 0, 1, 2])
        rm._ready = {Rendition(0)}
        rm._pending = []

        assert rm.is_switchable(2) is True
        assert rm.is_rendered(2) is False

    def test_false_outside_the_declared_range(self, test_prefs):
        rm = RenditionManager(test_prefs)
        rm._declared = _range_of([-2, -1, 0, 1, 2])

        assert rm.is_switchable(7) is False


class TestRenditionManagerEnsureSwitchable:
    """Tests for ensure_switchable."""

    def test_false_without_an_active_song(self, test_prefs):
        rm = RenditionManager(test_prefs)

        assert rm.ensure_switchable(2) is False

    def _write_rendition(self, tmp_path, fr, semitones_label, segments):
        (tmp_path / f"{fr.stream_uid}_audio_{semitones_label}.m3u8").write_text("#EXTM3U")
        for i in range(segments):
            (tmp_path / f"{fr.stream_uid}_audio_{semitones_label}_segment_{i:03d}.m4s").write_bytes(
                b"x"
            )

    def test_true_at_song_start_with_only_the_opening_rendered(self, test_prefs, tmp_path):
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._fr = fr
        rm._declared = _range_of([0, 1, 2])
        rm._pending = _range_of([1, 2])
        self._write_rendition(tmp_path, fr, "p2", segments=3)

        with patch.object(rm, "_launch_render"):
            assert rm.ensure_switchable(2, position=0, timeout=1) is True

    def test_starts_a_render_for_a_pitch_outside_the_pre_render_window(self, test_prefs, tmp_path):
        """Nothing is queued for it, so waiting alone would never succeed."""
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._fr = fr
        rm._declared = _range_of([0, 1, 2, 5])
        rm._pending = _range_of([1, 2])
        self._write_rendition(tmp_path, fr, "p5", segments=3)

        with patch.object(rm, "_launch_render") as mock_launch:
            assert rm.ensure_switchable(5, position=0, timeout=1) is True

        mock_launch.assert_called_once_with(fr, Rendition(5))

    def test_false_when_rendition_has_not_reached_the_playhead(self, test_prefs, tmp_path):
        """The opening being rendered is not enough to switch 60s into a song."""
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._fr = fr
        rm._declared = _range_of([0, 1, 2])
        rm._pending = _range_of([2])
        self._write_rendition(tmp_path, fr, "p2", segments=3)

        with patch.object(rm, "_launch_render"):
            assert rm.ensure_switchable(2, position=60, timeout=0.2) is False

    def test_true_when_rendition_covers_the_playhead(self, test_prefs, tmp_path):
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._fr = fr
        rm._declared = _range_of([0, 1, 2])
        rm._pending = _range_of([2])
        # 60s playhead + 6s lookahead over 3s segments needs 23 segments.
        self._write_rendition(tmp_path, fr, "p2", segments=23)

        with patch.object(rm, "_launch_render"):
            assert rm.ensure_switchable(2, position=60, timeout=1) is True

    def test_true_for_a_finished_rendition_near_the_end_of_a_song(self, test_prefs, tmp_path):
        """A completed render has no more segments coming, however late the playhead."""
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._fr = fr
        rm._declared = _range_of([0, 1, 2])
        self._write_rendition(tmp_path, fr, "p2", segments=4)
        finished = MagicMock()
        finished.poll.return_value = 0
        rm._audio_processes = {Rendition(2): finished}

        with patch.object(rm, "_launch_render"):
            assert rm.ensure_switchable(2, position=200, timeout=1) is True

    def test_false_when_segments_never_appear(self, test_prefs, tmp_path):
        rm = RenditionManager(test_prefs)
        rm._fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._declared = _range_of([0, 1, 2])
        rm._pending = _range_of([2])

        with patch.object(rm, "_launch_render"):
            assert rm.ensure_switchable(2, timeout=0.2) is False


class TestRenditionManagerReadiness:
    """Tests for is_rendered."""

    def test_is_rendered(self, test_prefs):
        rm = RenditionManager(test_prefs)
        rm._ready = {Rendition(0), Rendition(2)}

        assert rm.is_rendered(2) is True
        assert rm.is_rendered(5) is False
        assert rm.is_rendered(2, vocals_on=False) is False


class TestRenditionManagerKillAll:
    """Tests for kill_all tearing down every tracked process."""

    def test_terminates_video_and_all_audio_processes(self, test_prefs):
        rm = RenditionManager(test_prefs)
        video_proc = MagicMock()
        audio_proc1 = MagicMock()
        audio_proc2 = MagicMock()
        rm._video_process = video_proc
        rm._audio_processes = {Rendition(1): audio_proc1, Rendition(2): audio_proc2}
        rm._ready = {0, 1, 2}

        rm.kill_all()

        video_proc.terminate.assert_called_once()
        audio_proc1.terminate.assert_called_once()
        audio_proc2.terminate.assert_called_once()
        assert rm._audio_processes == {}
        assert rm._ready == set()
        assert rm._video_process is None

    def test_sets_stop_event(self, test_prefs):
        rm = RenditionManager(test_prefs)

        rm.kill_all()

        assert rm._stop_event.is_set()

    def test_safe_with_no_processes(self, test_prefs):
        rm = RenditionManager(test_prefs)

        rm.kill_all()  # should not raise


class TestVocalsOffRenditions:
    """The vocals-removed renditions, which exist only when a song has been separated."""

    @staticmethod
    def _started(test_prefs, tmp_path, separator, window=1):
        test_prefs.set("pitch_window_semitones", window)
        rm = RenditionManager(test_prefs, vocal_separator=separator)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        fr.file_path = "/songs/Song---abc12345678.mp4"
        with patch("pikaraoke.lib.rendition_manager._wait_until_ready", return_value=True):
            with patch("pikaraoke.lib.rendition_manager.build_video_only_ffmpeg_cmd"):
                with patch("pikaraoke.lib.rendition_manager.build_audio_only_ffmpeg_cmd"):
                    with patch.object(rm, "_render_remaining"):
                        rm.start(fr, base_semitones=0, source_path=fr.file_path)
        return rm

    def test_declared_alongside_vocals_on_when_separation_is_enabled(self, test_prefs, tmp_path):
        rm = self._started(test_prefs, tmp_path, _separator())

        pitches = range(MIN_SEMITONES, MAX_SEMITONES + 1)
        assert rm._declared == _range_of(pitches) + _range_of(pitches, vocals_on=False)

    def test_not_declared_when_separation_is_off(self, test_prefs, tmp_path):
        rm = self._started(test_prefs, tmp_path, _separator(mode="off"))

        assert all(r.vocals_on for r in rm._declared)

    def test_not_declared_without_a_separator(self, test_prefs, tmp_path):
        rm = self._started(test_prefs, tmp_path, None)

        assert all(r.vocals_on for r in rm._declared)

    def test_still_declared_while_the_song_is_being_separated(self, test_prefs, tmp_path):
        """The master playlist is read once, so a track that finishes later needs its slot now."""
        rm = self._started(test_prefs, tmp_path, _separator(track=None))

        assert Rendition(0, False) in rm._declared

    def test_same_pitch_without_vocals_is_rendered_ahead_of_the_next_pitch(
        self, test_prefs, tmp_path
    ):
        """A toggle is about as likely as a one-step key change, and both should be instant."""
        rm = self._started(test_prefs, tmp_path, _separator())

        assert rm._pending[0] == Rendition(0, False)
        assert rm._pending.index(Rendition(0, False)) < rm._pending.index(Rendition(1))

    def test_nothing_without_vocals_is_queued_before_its_track_exists(self, test_prefs, tmp_path):
        rm = self._started(test_prefs, tmp_path, _separator(track=None))

        assert all(r.vocals_on for r in rm._pending)

    def test_no_vocals_available_follows_the_separator_cache(self, test_prefs, tmp_path):
        separator = _separator(track=None)
        rm = self._started(test_prefs, tmp_path, separator)
        assert rm.no_vocals_available() is False

        separator.cached_track.return_value = "/songs/.stems/x.no_vocals.mp3"
        assert rm.no_vocals_available() is True

    def test_render_uses_the_separated_track_as_its_audio_source(self, test_prefs, tmp_path):
        rm = self._started(test_prefs, tmp_path, _separator(track="/stems/x.no_vocals.mp3"))
        fr = rm._fr

        with patch("pikaraoke.lib.rendition_manager.build_audio_only_ffmpeg_cmd") as build:
            rm._launch_render(fr, Rendition(2, False))

        assert build.call_args.args[-1] == "/stems/x.no_vocals.mp3"

    def test_vocals_on_render_uses_the_songs_own_audio(self, test_prefs, tmp_path):
        rm = self._started(test_prefs, tmp_path, _separator())
        fr = rm._fr

        with patch("pikaraoke.lib.rendition_manager.build_audio_only_ffmpeg_cmd") as build:
            rm._launch_render(fr, Rendition(2, True))

        assert build.call_args.args[-1] is None

    def test_render_is_refused_until_the_track_exists(self, test_prefs, tmp_path):
        rm = self._started(test_prefs, tmp_path, _separator(track=None))

        with patch("pikaraoke.lib.rendition_manager.build_audio_only_ffmpeg_cmd") as build:
            assert rm._launch_render(rm._fr, Rendition(2, False)) is None

        build.assert_not_called()

    def test_cannot_switch_to_vocals_off_before_its_track_exists(self, test_prefs, tmp_path):
        rm = self._started(test_prefs, tmp_path, _separator(track=None))

        assert rm.ensure_switchable(0, vocals_on=False, timeout=0.2) is False

    def test_switch_wait_looks_for_the_vocals_off_files(self, test_prefs, tmp_path):
        rm = self._started(test_prefs, tmp_path, _separator())
        fr = rm._fr
        (tmp_path / f"{fr.stream_uid}_audio_p0nv.m3u8").write_text("#EXTM3U")
        for i in range(3):
            (tmp_path / f"{fr.stream_uid}_audio_p0nv_segment_{i:03d}.m4s").write_bytes(b"x")

        with patch.object(rm, "_launch_render", return_value=MagicMock()):
            assert rm.ensure_switchable(0, vocals_on=False, position=0, timeout=1) is True

    def test_song_state_is_dropped_when_torn_down(self, test_prefs, tmp_path):
        rm = self._started(test_prefs, tmp_path, _separator())

        rm.kill_all()

        assert rm._source_path is None
        assert rm.no_vocals_available() is False
