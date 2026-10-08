"""Tests for how Karaoke drives vocal separation around queueing and playback."""

from unittest.mock import MagicMock

import pytest

from pikaraoke.karaoke import Karaoke

SONG = "/songs/Artist - Song---dQw4w9WgXcQ.mp4"


@pytest.fixture
def karaoke():
    """A stand-in carrying only what the separation methods touch."""
    k = MagicMock()
    k.separating_title = None
    k.song_manager.display_name_from_path.return_value = "Artist - Song"
    k.vocal_separator.cached_track.return_value = None
    return k


class TestSeparateQueuedSong:
    @pytest.mark.parametrize("mode", ["background", "before_play"])
    def test_queues_an_uncached_song(self, karaoke, mode):
        karaoke.vocal_separator.mode = mode
        Karaoke.separate_queued_song(karaoke, SONG)
        karaoke.vocal_separator.queue_separation.assert_called_once_with(SONG)

    def test_does_nothing_when_off(self, karaoke):
        karaoke.vocal_separator.mode = "off"
        Karaoke.separate_queued_song(karaoke, SONG)
        karaoke.vocal_separator.queue_separation.assert_not_called()


class TestPrepareVocals:
    def test_does_nothing_when_off(self, karaoke):
        karaoke.vocal_separator.mode = "off"
        Karaoke.prepare_vocals(karaoke, SONG)
        karaoke.vocal_separator.prioritize.assert_not_called()

    def test_does_nothing_for_a_song_already_separated(self, karaoke):
        karaoke.vocal_separator.mode = "before_play"
        karaoke.vocal_separator.cached_track.return_value = "/songs/.stems/x.no_vocals.mp3"
        Karaoke.prepare_vocals(karaoke, SONG)
        karaoke.vocal_separator.prioritize.assert_not_called()
        karaoke.vocal_separator.wait_for_head_start.assert_not_called()

    def test_background_mode_puts_the_song_first_but_does_not_wait(self, karaoke):
        """Its vocals-off track has to keep ahead of playback, so no other song goes first."""
        karaoke.vocal_separator.mode = "background"
        Karaoke.prepare_vocals(karaoke, SONG)
        karaoke.vocal_separator.prioritize.assert_called_once_with(SONG)
        karaoke.vocal_separator.wait_for_head_start.assert_not_called()

    def test_before_play_waits_for_a_head_start_and_shows_why(self, karaoke):
        karaoke.vocal_separator.mode = "before_play"
        shown = []
        karaoke.vocal_separator.wait_for_head_start.side_effect = lambda path: shown.append(
            karaoke.separating_title
        )

        Karaoke.prepare_vocals(karaoke, SONG)

        karaoke.vocal_separator.prioritize.assert_called_once_with(SONG)
        assert shown == ["Artist - Song"]
        assert karaoke.separating_title is None
        # Once to show the message, once to take it down.
        assert karaoke.update_now_playing_socket.call_count == 2

    def test_clears_the_message_even_if_waiting_fails(self, karaoke):
        """A stuck message on the splash screen would outlive the failed song."""
        karaoke.vocal_separator.mode = "before_play"
        karaoke.vocal_separator.wait_for_head_start.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            Karaoke.prepare_vocals(karaoke, SONG)

        assert karaoke.separating_title is None


class TestSetVocals:
    @pytest.fixture
    def karaoke(self):
        k = MagicMock()
        k.playback_controller.now_playing_transpose = 2
        k.playback_controller.can_switch_vocals.return_value = True
        k.playback_controller.prepare_switch.return_value = True
        return k

    def test_switches_vocals_off_at_the_current_pitch(self, karaoke):
        assert Karaoke.set_vocals(karaoke, False) is True

        karaoke.playback_controller.prepare_switch.assert_called_once_with(2, False)
        assert karaoke.playback_controller.now_playing_vocals is False

    def test_tells_the_players_so_the_splash_switches_track(self, karaoke):
        Karaoke.set_vocals(karaoke, False)

        karaoke.events.emit.assert_called_with("now_playing_update")

    def test_refuses_when_the_song_cannot_switch_yet(self, karaoke):
        """Before its vocals are separated: the state must not flip to one nothing plays."""
        karaoke.playback_controller.can_switch_vocals.return_value = False
        karaoke.playback_controller.now_playing_vocals = True

        assert Karaoke.set_vocals(karaoke, False) is False

        assert karaoke.playback_controller.now_playing_vocals is True
        karaoke.playback_controller.prepare_switch.assert_not_called()
        karaoke.events.emit.assert_not_called()

    def test_refuses_while_the_vocals_off_track_is_still_catching_up(self, karaoke):
        """Separation can still be behind the playhead early in a song; switching to a
        track with nothing there yet would stall the player."""
        karaoke.playback_controller.prepare_switch.return_value = False
        karaoke.playback_controller.now_playing_vocals = True

        assert Karaoke.set_vocals(karaoke, False) is False

        assert karaoke.playback_controller.now_playing_vocals is True
        karaoke.events.emit.assert_not_called()
