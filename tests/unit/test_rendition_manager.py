"""Unit tests for rendition_manager module."""

from unittest.mock import MagicMock, call, patch

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

        with patch.object(rm, "_pace_renditions"):
            result = rm.start(fr, base_semitones=0)

        assert result.success is True
        assert result.stream_url == f"/stream/{fr.stream_uid}.m3u8"
        assert (tmp_path / f"{fr.stream_uid}.m3u8").exists()
        # The base pitch is what's being listened to from the first frame,
        # so it starts out as the one holding foreground priority.
        assert rm._active == 0

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

        with patch.object(rm, "_pace_renditions"):
            rm.start(fr, base_semitones=0)

        assert rm._declared == list(range(MIN_SEMITONES, MAX_SEMITONES + 1))
        # Nearest-to-base first, so the pitches a singer is most likely to
        # step to next are ready soonest.
        assert rm._window == [0, -1, 1, -2, 2]

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

        with patch.object(rm, "_pace_renditions"):
            rm.start(fr, base_semitones=4)

        assert max(rm._window) == MAX_SEMITONES
        assert min(rm._window) == -1


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
        self._write_rendition(tmp_path, fr, "p2", segments=3)

        with patch.object(rm, "_launch_render"):
            assert rm.ensure_switchable(2, position=0, timeout=1) is True

    def test_starts_a_render_for_a_pitch_outside_the_pre_render_window(self, test_prefs, tmp_path):
        """Nothing is queued for it, so waiting alone would never succeed."""
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._fr = fr
        rm._declared = [0, 1, 2, 5]
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
        self._write_rendition(tmp_path, fr, "p2", segments=3)

        with patch.object(rm, "_launch_render"):
            assert rm.ensure_switchable(2, position=60, timeout=0.2) is False

    def test_true_when_rendition_covers_the_playhead(self, test_prefs, tmp_path):
        rm = RenditionManager(test_prefs)
        fr = _make_mock_fr(tmp_dir=str(tmp_path))
        rm._fr = fr
        rm._declared = [0, 1, 2]
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
            patch.object(rm, "_pace_renditions"),
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


class TestRenditionManagerActivePitch:
    """Tests for _set_active: exactly one pitch holds foreground priority."""

    def test_active_pitch_is_restored_and_the_rest_are_lowered(self, test_prefs):
        rm = RenditionManager(test_prefs)
        procs = {s: MagicMock(**{"poll.return_value": None}) for s in (-1, 0, 1)}
        rm._audio_processes = dict(procs)

        with (
            patch("pikaraoke.lib.rendition_manager.lower_priority") as mock_lower,
            patch("pikaraoke.lib.rendition_manager.restore_priority") as mock_restore,
        ):
            rm._set_active(1)

        mock_restore.assert_called_once_with(procs[1])
        assert {c.args[0] for c in mock_lower.call_args_list} == {procs[-1], procs[0]}
        assert rm._active == 1

    def test_stepping_through_keys_leaves_only_the_last_one_foreground(self, test_prefs):
        """The bug this exists for: every pitch stepped through used to stay
        at foreground priority, so six quick steps left six rivals."""
        rm = RenditionManager(test_prefs)
        procs = {s: MagicMock(**{"poll.return_value": None}) for s in range(-2, 3)}
        rm._audio_processes = dict(procs)

        with (
            patch("pikaraoke.lib.rendition_manager.lower_priority") as mock_lower,
            patch("pikaraoke.lib.rendition_manager.restore_priority") as mock_restore,
        ):
            for pitch in (-1, -2, -1, 0, 1, 2):
                rm._set_active(pitch)

        # Only the final pitch is foreground; the five it passed through are
        # all back down, each demoted on the step that superseded it.
        assert rm._active == 2
        assert mock_restore.call_args_list[-1].args[0] is procs[2]
        assert procs[1] in {c.args[0] for c in mock_lower.call_args_list}

    def test_skips_a_process_that_already_exited(self, test_prefs):
        rm = RenditionManager(test_prefs)
        finished = MagicMock(**{"poll.return_value": 0})
        running = MagicMock(**{"poll.return_value": None})
        rm._audio_processes = {0: finished, 1: running}

        with (
            patch("pikaraoke.lib.rendition_manager.lower_priority") as mock_lower,
            patch("pikaraoke.lib.rendition_manager.restore_priority") as mock_restore,
        ):
            rm._set_active(1)

        mock_restore.assert_called_once_with(running)
        mock_lower.assert_not_called()

    def test_kill_all_clears_the_active_pitch(self, test_prefs):
        rm = RenditionManager(test_prefs)
        rm._audio_processes = {2: MagicMock(**{"poll.return_value": None})}
        rm._set_active(2)

        rm.kill_all()

        assert rm._active is None


class TestRenditionManagerPacing:
    """Tests for _pace_once: renditions are kept a bounded distance past the
    playhead and frozen the rest of the time, one running at a time."""

    @staticmethod
    def _rm_with_window(prefs, window, rendered):
        """A manager mid-song, with `rendered` seconds of audio per pitch."""
        rm = RenditionManager(prefs)
        rm._window = list(window)
        rm._audio_processes = {s: MagicMock(**{"poll.return_value": None}) for s in rendered}
        rm._rendering = set(rendered)
        rm._rendered_seconds = lambda fr, s: rendered[s]
        return rm

    def test_a_rendition_far_enough_ahead_is_frozen(self, test_prefs, tmp_path):
        rm = self._rm_with_window(test_prefs, [0, 1], {0: 300, 1: 300})
        rm._position = 10

        with (
            patch.object(rm, "_suspend") as mock_suspend,
            patch.object(rm, "_resume") as mock_resume,
        ):
            rm._pace_once(_make_mock_fr(tmp_dir=str(tmp_path)))

        assert {c.args[0] for c in mock_suspend.call_args_list} == {0, 1}
        mock_resume.assert_not_called()

    def test_only_one_rendition_runs_at_a_time(self, test_prefs, tmp_path):
        """The whole point of pacing: three pitches all behind their target
        must not all run at once, or they starve each other."""
        rm = self._rm_with_window(test_prefs, [0, 1, -1], {0: 0, 1: 0, -1: 0})
        rm._position = 60

        with (
            patch.object(rm, "_suspend") as mock_suspend,
            patch.object(rm, "_resume") as mock_resume,
        ):
            rm._pace_once(_make_mock_fr(tmp_dir=str(tmp_path)))

        assert len(mock_resume.call_args_list) == 1
        assert {c.args[0] for c in mock_suspend.call_args_list} == {1, -1}

    def test_the_playing_pitch_gets_the_slot_first(self, test_prefs, tmp_path):
        """Whatever is being listened to must never fall behind the playhead,
        so it outranks pitches nobody is hearing yet."""
        rm = self._rm_with_window(test_prefs, [0, 1, 2], {0: 0, 1: 0, 2: 0})
        rm._position = 60
        rm._active = 2

        with (
            patch.object(rm, "_suspend"),
            patch.object(rm, "_resume") as mock_resume,
        ):
            rm._pace_once(_make_mock_fr(tmp_dir=str(tmp_path)))

        mock_resume.assert_called_once_with(2)

    def test_a_pitch_someone_is_waiting_on_keeps_running(self, test_prefs, tmp_path):
        """A caller blocked in ensure_switchable needs its target making
        progress, however far ahead of the playhead it already is."""
        rm = self._rm_with_window(test_prefs, [0, 1], {0: 0, 1: 300})
        rm._position = 10
        rm._awaited = 1

        with (
            patch.object(rm, "_suspend") as mock_suspend,
            patch.object(rm, "_resume") as mock_resume,
        ):
            rm._pace_once(_make_mock_fr(tmp_dir=str(tmp_path)))

        assert 1 in {c.args[0] for c in mock_resume.call_args_list}
        assert 1 not in {c.args[0] for c in mock_suspend.call_args_list}

    def test_an_unstarted_pitch_is_launched(self, test_prefs, tmp_path):
        rm = self._rm_with_window(test_prefs, [0, 1], {0: 300})
        rm._rendered_seconds = lambda fr, s: 300 if s == 0 else 0
        rm._position = 10

        with (
            patch.object(rm, "_suspend"),
            patch.object(rm, "_launch_render") as mock_launch,
        ):
            rm._pace_once(_make_mock_fr(tmp_dir=str(tmp_path)))

        mock_launch.assert_called_once()
        assert mock_launch.call_args.args[1] == 1

    def test_a_finished_rendition_is_left_alone(self, test_prefs, tmp_path):
        """Nothing to pace once ffmpeg has written the whole song."""
        rm = self._rm_with_window(test_prefs, [0], {0: 30})
        rm._audio_processes[0].poll.return_value = 0
        rm._position = 600

        with (
            patch.object(rm, "_suspend") as mock_suspend,
            patch.object(rm, "_resume") as mock_resume,
        ):
            rm._pace_once(_make_mock_fr(tmp_dir=str(tmp_path)))

        mock_resume.assert_not_called()
        assert mock_suspend.call_args_list == [call(0)]

    def test_lead_is_measured_from_the_playhead_not_the_start(self, test_prefs, tmp_path):
        """Content that was plenty at the start of the song stops being
        enough once the playhead has moved past it."""
        rm = self._rm_with_window(test_prefs, [0], {0: 40})
        fr = _make_mock_fr(tmp_dir=str(tmp_path))

        rm._position = 0
        with patch.object(rm, "_suspend") as early_suspend, patch.object(rm, "_resume"):
            rm._pace_once(fr)

        rm._position = 120
        with patch.object(rm, "_suspend"), patch.object(rm, "_resume") as late_resume:
            rm._pace_once(fr)

        assert early_suspend.call_args_list == [call(0)]
        late_resume.assert_called_once_with(0)


class TestRenditionManagerSuspendResume:
    """Tests for the suspend/resume bookkeeping."""

    def test_suspend_freezes_a_running_render_once(self, test_prefs):
        rm = RenditionManager(test_prefs)
        proc = MagicMock(**{"poll.return_value": None})
        rm._audio_processes = {1: proc}

        with patch("pikaraoke.lib.rendition_manager.suspend_process") as mock_suspend:
            rm._suspend(1)
            rm._suspend(1)

        mock_suspend.assert_called_once_with(proc)
        assert rm._suspended == {1}

    def test_resume_thaws_only_a_suspended_render(self, test_prefs):
        rm = RenditionManager(test_prefs)
        proc = MagicMock(**{"poll.return_value": None})
        rm._audio_processes = {1: proc}

        with patch("pikaraoke.lib.rendition_manager.resume_process") as mock_resume:
            rm._resume(1)  # not suspended: nothing to do
            rm._suspended.add(1)
            rm._resume(1)

        mock_resume.assert_called_once_with(proc)
        assert rm._suspended == set()

    def test_asking_for_a_suspended_pitch_resumes_it(self, test_prefs, tmp_path):
        """ensure_switchable's launch path has to thaw a frozen rendition -
        a caller waiting on one that stays frozen would wait forever."""
        rm = RenditionManager(test_prefs)
        proc = MagicMock(**{"poll.return_value": None})
        rm._audio_processes = {1: proc}
        rm._rendering = {1}
        rm._suspended = {1}

        with patch("pikaraoke.lib.rendition_manager.resume_process") as mock_resume:
            rm._launch_render(_make_mock_fr(tmp_dir=str(tmp_path)), 1, background=False)

        mock_resume.assert_called_once_with(proc)
        assert rm._suspended == set()
