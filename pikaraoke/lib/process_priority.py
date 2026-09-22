"""OS-level scheduling priority helpers for background subprocess work.

Used to keep speculative background work (downloads, pitch pre-rendering)
from competing with foreground work (live playback, a render someone is
actively waiting on) for CPU and disk I/O on constrained hardware like a
Raspberry Pi.
"""

from __future__ import annotations

import logging
import subprocess

import psutil

from pikaraoke.lib.get_platform import is_windows


def lower_priority(process: subprocess.Popen) -> None:
    """Drop a process to background priority so it never competes with foreground work.

    Priority is inherited, which is what covers any child processes it spawns.
    """
    try:
        child = psutil.Process(process.pid)
        if is_windows():
            child.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            child.ionice(psutil.IOPRIO_VERYLOW)
        else:
            child.nice(10)
            # macOS has no ionice at all, hence AttributeError below.
            child.ionice(psutil.IOPRIO_CLASS_IDLE)
    except (psutil.Error, AttributeError, NotImplementedError, OSError) as e:
        logging.debug(f"Could not lower process priority: {e}")


def restore_priority(process: subprocess.Popen) -> None:
    """Return a process to normal scheduling priority.

    For a process lower_priority already dropped, e.g. a background render
    that someone is now actively waiting on.
    """
    try:
        child = psutil.Process(process.pid)
        if is_windows():
            child.nice(psutil.NORMAL_PRIORITY_CLASS)
            child.ionice(psutil.IOPRIO_NORMAL)
        else:
            child.nice(0)
            child.ionice(psutil.IOPRIO_CLASS_BE)
    except (psutil.Error, AttributeError, NotImplementedError, OSError) as e:
        logging.debug(f"Could not restore process priority: {e}")
