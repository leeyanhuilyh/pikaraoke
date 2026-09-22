"""Unit tests for process_priority module."""

from unittest.mock import MagicMock, patch

from pikaraoke.lib.process_priority import lower_priority, restore_priority


class TestLowerPriority:
    """Tests for lower_priority."""

    @patch("pikaraoke.lib.process_priority.is_windows", return_value=False)
    @patch("pikaraoke.lib.process_priority.psutil.Process")
    def test_posix_drops_nice_and_ionice(self, mock_process_cls, mock_is_windows):
        mock_child = MagicMock()
        mock_process_cls.return_value = mock_child

        lower_priority(MagicMock(pid=123))

        mock_child.nice.assert_called_once_with(10)
        mock_child.ionice.assert_called_once()

    @patch("pikaraoke.lib.process_priority.psutil.IOPRIO_VERYLOW", 0, create=True)
    @patch("pikaraoke.lib.process_priority.psutil.BELOW_NORMAL_PRIORITY_CLASS", 0, create=True)
    @patch("pikaraoke.lib.process_priority.is_windows", return_value=True)
    @patch("pikaraoke.lib.process_priority.psutil.Process")
    def test_windows_branch_sets_nice_and_ionice(self, mock_process_cls, mock_is_windows):
        # The real Windows-only psutil constants only exist when psutil is
        # imported on Windows, so this stands them in with create=True and
        # only checks the branch was taken and called both, not the values.
        mock_child = MagicMock()
        mock_process_cls.return_value = mock_child

        lower_priority(MagicMock(pid=123))

        mock_child.nice.assert_called_once()
        mock_child.ionice.assert_called_once()

    @patch("pikaraoke.lib.process_priority.psutil.Process")
    def test_swallows_errors_for_a_process_that_already_exited(self, mock_process_cls):
        import psutil

        mock_process_cls.side_effect = psutil.NoSuchProcess(123)

        lower_priority(MagicMock(pid=123))  # should not raise

    @patch("pikaraoke.lib.process_priority.is_windows", return_value=False)
    @patch("pikaraoke.lib.process_priority.psutil.Process")
    def test_swallows_missing_ionice_on_macos(self, mock_process_cls, mock_is_windows):
        mock_child = MagicMock()
        mock_child.ionice.side_effect = AttributeError
        mock_process_cls.return_value = mock_child

        lower_priority(MagicMock(pid=123))  # should not raise


class TestRestorePriority:
    """Tests for restore_priority."""

    @patch("pikaraoke.lib.process_priority.is_windows", return_value=False)
    @patch("pikaraoke.lib.process_priority.psutil.Process")
    def test_posix_restores_normal_nice_and_ionice(self, mock_process_cls, mock_is_windows):
        mock_child = MagicMock()
        mock_process_cls.return_value = mock_child

        restore_priority(MagicMock(pid=123))

        mock_child.nice.assert_called_once_with(0)
        mock_child.ionice.assert_called_once()

    @patch("pikaraoke.lib.process_priority.psutil.Process")
    def test_swallows_errors_for_a_process_that_already_exited(self, mock_process_cls):
        import psutil

        mock_process_cls.side_effect = psutil.NoSuchProcess(123)

        restore_priority(MagicMock(pid=123))  # should not raise
