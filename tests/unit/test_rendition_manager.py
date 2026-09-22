"""Unit tests for rendition_manager module."""

from unittest.mock import MagicMock, patch

import pytest

from pikaraoke.lib.preference_manager import PreferenceManager
from pikaraoke.lib.rendition_manager import (
    MAX_SEMITONES,
    MIN_SEMITONES,
    RenditionManager,
    audio_playlist_path,
    build_master_playlist,
    semitone_label,
    video_playlist_path,
)


@pytest.fixture
def test_prefs():
    """Create a PreferenceManager for testing."""
    return PreferenceManager("/nonexistent/test_config.ini")


@pytest.fixture(autouse=True)
def no_reprioritize():
    """Keep psutil away from whatever pid a mocked Popen invents."""
    with (
        patch("pikaraoke.lib.rendition_manager.lower_priority"),
        patch("pikaraoke.lib.rendition_manager.restore_priority"),
    ):
        yield


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
        assert audio_playlist_path(fr, -3) == "/tmp/12345_audio_m3.m3u8"


class TestBuildMasterPlaylist:
    """Tests for build_master_playlist."""

    def test_declares_one_media_entry_per_offset(self):
        fr = _make_mock_fr()
        playlist = build_master_playlist(fr, [-1, 0, 1], base_semitones=0)

        assert playlist.count("#EXT-X-MEDIA:TYPE=AUDIO") == 3
        assert playlist.count("#EXT-X-STREAM-INF") == 1
        assert "12345_video.m3u8" in playlist

    def test_default_flag_on_base_semitone_only(self):
        fr = _make_mock_fr()
        playlist = build_master_playlist(fr, [-1, 0, 1], base_semitones=1)

        default_lines = [line for line in playlist.splitlines() if "DEFAULT=YES" in line]
        assert len(default_lines) == 1
        assert 'NAME="1"' in default_lines[0]

    def test_every_pitch_gets_a_distinct_uri(self):
        fr = _make_mock_fr()
        offsets = list(range(-6, 7))
        playlist = build_master_playlist(fr, offsets, base_semitones=0)

        uris = [line.split('URI="')[1] for line in playlist.splitlines() if 'URI="' in line]
        assert len(uris) == len(offsets)
        assert len(set(uris)) == len(offsets)


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

    @patch("pikaraoke.lib.rendition_manager._wait_until_ready", return_value=True)
    @patch("pikaraoke.lib.rendition_manager.build_audio_only_ffmpeg_cmd")
    @patch("pikaraoke.lib.rendition_manager.build_video_only_ffmpeg_cmd")
    def test_window_renders_before_start_returns(
        self, mock_video_cmd, mock_audio_cmd, mock_ready, test_prefs, tmp_path
    ):
        """BLOCK_PLAYBACK_UNTIL_WINDOW_RENDERED: the whole window is done,
        not just queued, by the time start() hands back a stream URL."""
        test_prefs.set("pitch_window_semitones", 2)
        mock_video_cmd.return_value.run_async.return_value = MagicMock()
        mock_audio_cmd.return_value.run_async.return_value = MagicMock()
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))

        result = rm.start(fr, base_semitones=0)

        assert result.success is True
        assert rm._pending == []
        assert sorted(rm._ready) == [-2, -1, 0, 1, 2]
        assert rm._bg_thread is None

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

        assert rm._declared == list(range(MIN_SEMITONES, MAX_SEMITONES + 1))
        # Nearest-to-base first, so the pitches a singer is most likely to
        # step to next are ready soonest.
        assert rm._pending == [-1, 1, -2, 2]

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

        assert max(rm._pending) == MAX_SEMITONES
        assert min(rm._pending) == -1


class TestRenditionManagerConcurrencyCap:
    """Tests for MAX_CONCURRENT_BACKGROUND_RENDERS."""

    def test_slot_unavailable_when_cap_reached(self, test_prefs):
        rm = RenditionManager(test_prefs)
        alive1, alive2 = MagicMock(), MagicMock()
        alive1.poll.return_value = None
        alive2.poll.return_value = None
        rm._audio_processes = {1: alive1, 2: alive2}
        rm._background_alive = {1, 2}

        assert rm._background_slot_available() is False

    def test_slot_available_below_cap(self, test_prefs):
        rm = RenditionManager(test_prefs)
        alive1 = MagicMock()
        alive1.poll.return_value = None
        rm._audio_processes = {1: alive1}
        rm._background_alive = {1}

        assert rm._background_slot_available() is True

    def test_prunes_a_render_that_finished(self, test_prefs):
        rm = RenditionManager(test_prefs)
        finished = MagicMock()
        finished.poll.return_value = 0
        rm._audio_processes = {1: finished}
        rm._background_alive = {1}

        assert rm._background_slot_available() is True
        assert rm._background_alive == set()

    def test_a_boosted_render_no_longer_counts_against_the_cap(self, test_prefs, tmp_path):
        """Someone actively waiting on a pitch isn't background contention."""
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        proc1, proc2 = MagicMock(), MagicMock()
        proc1.poll.return_value = None
        proc2.poll.return_value = None
        rm._audio_processes = {1: proc1, 2: proc2}
        rm._rendering = {1, 2}
        rm._background_alive = {1, 2}

        rm._launch_render(fr, 1, background=False)  # boost, doesn't spawn (already running)

        assert rm._background_alive == {2}
        assert rm._background_slot_available() is True

    def test_third_render_waits_for_a_slot_to_free_up(self, test_prefs, tmp_path):
        """Renders 1 and 2 fill the cap; 3 must wait until one exits."""
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._pending = [1, 2, 3]
        alive_count_at_launch = []
        poll_calls = {"n": 0}

        def fake_render_one(fr_arg, semitones):
            alive_count_at_launch.append(len(rm._background_alive))
            proc = MagicMock()
            if semitones == 1:
                # Exits only after being polled a few times, simulating a
                # background render that keeps running for a while.
                def poll():
                    poll_calls["n"] += 1
                    return 0 if poll_calls["n"] > 2 else None

                proc.poll.side_effect = poll
            else:
                proc.poll.return_value = None
            rm._audio_processes[semitones] = proc
            rm._background_alive.add(semitones)
            return True

        with patch.object(rm, "_render_one", side_effect=fake_render_one):
            rm._render_remaining(fr)

        assert alive_count_at_launch == [0, 1, 1]
        assert poll_calls["n"] > 2


class TestRenditionManagerSequentialRendering:
    """Tests for _render_remaining's queue draining."""

    def test_renders_queue_in_order(self, test_prefs, tmp_path):
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._pending = [1, 2, 3]
        call_order = []

        def fake_render_one(fr_arg, semitones):
            call_order.append(semitones)
            return True

        with patch.object(rm, "_render_one", side_effect=fake_render_one):
            rm._render_remaining(fr)

        assert call_order == [1, 2, 3]

    def test_advances_once_a_rendition_is_ready_not_once_it_fully_finishes(
        self, test_prefs, tmp_path
    ):
        """A rendition keeps encoding the rest of the song in the background
        after it's ready - the queue must not wait for that to finish, or
        each item ends up gated on close to the song's full runtime."""
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._pending = [1, 2]
        events = []

        def fake_render_one(fr_arg, semitones):
            events.append(f"ready-{semitones}")
            proc = MagicMock()
            proc.wait.side_effect = lambda: events.append(f"finished-{semitones}")
            rm._audio_processes[semitones] = proc
            return True

        with patch.object(rm, "_render_one", side_effect=fake_render_one):
            rm._render_remaining(fr)

        # Both renditions are started (ready) before either is waited on to
        # fully finish - "finished" events, if any, must not appear between them.
        assert events == ["ready-1", "ready-2"]

    def test_stops_when_stop_event_set_mid_sequence(self, test_prefs, tmp_path):
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._pending = [1, 2, 3]
        call_order = []

        def fake_render_one(fr_arg, semitones):
            call_order.append(semitones)
            if semitones == 1:
                rm._stop_event.set()
            return True

        with patch.object(rm, "_render_one", side_effect=fake_render_one):
            rm._render_remaining(fr)

        assert call_order == [1]

    def test_prioritize_moves_a_queued_pitch_to_the_front(self, test_prefs, tmp_path):
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._pending = [1, -1, 2, -2]
        call_order = []

        def fake_render_one(fr_arg, semitones):
            call_order.append(semitones)
            return True

        rm.prioritize(-2)

        with patch.object(rm, "_render_one", side_effect=fake_render_one):
            rm._render_remaining(fr)

        assert call_order == [-2, 1, -1, 2]

    def test_prioritize_ignores_a_pitch_that_is_not_queued(self, test_prefs):
        rm = RenditionManager(test_prefs)
        rm._pending = [1, 2]

        rm.prioritize(9)

        assert rm._pending == [1, 2]


class TestRenditionManagerIsSwitchable:
    """Tests for is_switchable, which decides switch vs restart."""

    def test_true_for_a_declared_pitch(self, test_prefs):
        rm = RenditionManager(test_prefs)
        rm._declared = [-2, -1, 0, 1, 2]

        assert rm.is_switchable(2) is True

    def test_true_even_when_not_rendered_or_pre_rendered(self, test_prefs):
        """The master playlist advertises it, so the player can switch to it."""
        rm = RenditionManager(test_prefs)
        rm._declared = [-2, -1, 0, 1, 2]
        rm._ready = {0}
        rm._pending = []

        assert rm.is_switchable(2) is True
        assert rm.is_rendered(2) is False

    def test_false_outside_the_declared_range(self, test_prefs):
        rm = RenditionManager(test_prefs)
        rm._declared = [-2, -1, 0, 1, 2]

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
        rm._declared = [0, 1, 2]
        rm._pending = [1, 2]
        self._write_rendition(tmp_path, fr, "p2", segments=3)

        with patch.object(rm, "_launch_render"):
            assert rm.ensure_switchable(2, position=0, timeout=1) is True

    def test_starts_a_render_for_a_pitch_outside_the_pre_render_window(self, test_prefs, tmp_path):
        """Nothing is queued for it, so waiting alone would never succeed."""
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._fr = fr
        rm._declared = [0, 1, 2, 5]
        rm._pending = [1, 2]
        self._write_rendition(tmp_path, fr, "p5", segments=3)

        with patch.object(rm, "_launch_render") as mock_launch:
            assert rm.ensure_switchable(5, position=0, timeout=1) is True

        mock_launch.assert_called_once_with(fr, 5, background=False)

    def test_false_when_rendition_has_not_reached_the_playhead(self, test_prefs, tmp_path):
        """The opening being rendered is not enough to switch 60s into a song."""
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._fr = fr
        rm._declared = [0, 1, 2]
        rm._pending = [2]
        self._write_rendition(tmp_path, fr, "p2", segments=3)

        with patch.object(rm, "_launch_render"):
            assert rm.ensure_switchable(2, position=60, timeout=0.2) is False

    def test_true_when_rendition_covers_the_playhead(self, test_prefs, tmp_path):
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._fr = fr
        rm._declared = [0, 1, 2]
        rm._pending = [2]
        # 60s playhead + 6s lookahead over 3s segments needs 23 segments.
        self._write_rendition(tmp_path, fr, "p2", segments=23)

        with patch.object(rm, "_launch_render"):
            assert rm.ensure_switchable(2, position=60, timeout=1) is True

    def test_true_for_a_finished_rendition_near_the_end_of_a_song(self, test_prefs, tmp_path):
        """A completed render has no more segments coming, however late the playhead."""
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._fr = fr
        rm._declared = [0, 1, 2]
        self._write_rendition(tmp_path, fr, "p2", segments=4)
        finished = MagicMock()
        finished.poll.return_value = 0
        rm._audio_processes = {2: finished}

        with patch.object(rm, "_launch_render"):
            assert rm.ensure_switchable(2, position=200, timeout=1) is True

    def test_false_when_segments_never_appear(self, test_prefs, tmp_path):
        rm = RenditionManager(test_prefs)
        rm._fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._declared = [0, 1, 2]
        rm._pending = [2]

        with patch.object(rm, "_launch_render"):
            assert rm.ensure_switchable(2, timeout=0.2) is False


class TestRenditionManagerReadiness:
    """Tests for is_rendered."""

    def test_is_rendered(self, test_prefs):
        rm = RenditionManager(test_prefs)
        rm._ready = {0, 2}

        assert rm.is_rendered(2) is True
        assert rm.is_rendered(5) is False


class TestRenditionManagerPriority:
    """Tests for OS scheduling priority: background renders never outrank
    live playback or a render someone is actively waiting on."""

    @patch("pikaraoke.lib.rendition_manager.build_audio_only_ffmpeg_cmd")
    def test_background_render_is_lowered(self, mock_cmd, test_prefs, tmp_path):
        mock_cmd.return_value.run_async.return_value = MagicMock()
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))

        with (
            patch("pikaraoke.lib.rendition_manager.lower_priority") as mock_lower,
            patch("pikaraoke.lib.rendition_manager.restore_priority") as mock_restore,
        ):
            rm._launch_render(fr, 3, background=True)

        mock_lower.assert_called_once()
        mock_restore.assert_not_called()

    @patch("pikaraoke.lib.rendition_manager.build_audio_only_ffmpeg_cmd")
    def test_on_demand_render_is_kept_at_normal_priority(self, mock_cmd, test_prefs, tmp_path):
        mock_cmd.return_value.run_async.return_value = MagicMock()
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))

        with (
            patch("pikaraoke.lib.rendition_manager.lower_priority") as mock_lower,
            patch("pikaraoke.lib.rendition_manager.restore_priority") as mock_restore,
        ):
            rm._launch_render(fr, 3, background=False)

        mock_restore.assert_called_once()
        mock_lower.assert_not_called()

    @patch("pikaraoke.lib.rendition_manager.build_audio_only_ffmpeg_cmd")
    def test_already_running_background_render_is_boosted_on_demand(
        self, mock_cmd, test_prefs, tmp_path
    ):
        """A pitch already mid-render in the background queue, once someone
        actually asks for it, must stop being deprioritized - the caller is
        waiting on it now, not speculating."""
        mock_cmd.return_value.run_async.return_value = MagicMock()
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))

        with (
            patch("pikaraoke.lib.rendition_manager.lower_priority"),
            patch("pikaraoke.lib.rendition_manager.restore_priority"),
        ):
            rm._launch_render(fr, 3, background=True)  # started by the background queue

        with (
            patch("pikaraoke.lib.rendition_manager.lower_priority") as mock_lower,
            patch("pikaraoke.lib.rendition_manager.restore_priority") as mock_restore,
        ):
            proc = rm._launch_render(fr, 3, background=False)  # now requested on demand

        mock_restore.assert_called_once_with(proc)
        mock_lower.assert_not_called()
        mock_cmd.return_value.run_async.assert_called_once()  # no second process spawned

    @patch("pikaraoke.lib.rendition_manager.build_video_only_ffmpeg_cmd")
    @patch("pikaraoke.lib.rendition_manager.build_audio_only_ffmpeg_cmd")
    @patch("pikaraoke.lib.rendition_manager._wait_until_ready", return_value=True)
    def test_base_pitch_render_at_song_start_is_not_lowered(
        self, mock_ready, mock_audio_cmd, mock_video_cmd, test_prefs, tmp_path
    ):
        """The base pitch is needed immediately to start playback, not speculative."""
        mock_video_cmd.return_value.run_async.return_value = MagicMock()
        mock_audio_cmd.return_value.run_async.return_value = MagicMock()
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))

        with (
            patch.object(rm, "_render_remaining"),
            patch("pikaraoke.lib.rendition_manager.lower_priority") as mock_lower,
            patch("pikaraoke.lib.rendition_manager.restore_priority") as mock_restore,
        ):
            result = rm.start(fr, base_semitones=0)

        assert result.success is True
        mock_lower.assert_not_called()
        mock_restore.assert_called_once()


class TestRenditionManagerKillAll:
    """Tests for kill_all tearing down every tracked process."""

    def test_terminates_video_and_all_audio_processes(self, test_prefs):
        rm = RenditionManager(test_prefs)
        video_proc = MagicMock()
        audio_proc1 = MagicMock()
        audio_proc2 = MagicMock()
        rm._video_process = video_proc
        rm._audio_processes = {1: audio_proc1, 2: audio_proc2}
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
