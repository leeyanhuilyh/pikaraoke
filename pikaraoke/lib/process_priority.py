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


def suspend_process(process: subprocess.Popen) -> None:
    """Freeze a process outright, so it consumes no CPU until resumed.

    For work that is far enough ahead to be put down entirely rather than
    merely deprioritized: a deprioritized process still takes CPU whenever
    a core is free. The process keeps its state and picks up exactly where
    it left off, so this is not the same as killing and restarting it.
    """
    _suspend_or_resume(process, "suspend")


def resume_process(process: subprocess.Popen) -> None:
    """Let a suspended process continue from where it was frozen.

    Callers rely on this not raising: teardown resumes a process before
    terminating it, because a suspended process would otherwise sit out its
    SIGTERM, and a raise here would leave it running instead of killed.
    """
    _suspend_or_resume(process, "resume")


def _suspend_or_resume(process: subprocess.Popen, action: str) -> None:
    """Freeze or thaw a process, swallowing anything psutil objects to.

    ValueError and TypeError cover a pid psutil rejects outright, worth
    ignoring for the same reason as an already-exited process: either way
    there is nothing left to stop or start.
    """
    try:
        getattr(psutil.Process(process.pid), action)()
    except (psutil.Error, OSError, ValueError, TypeError) as e:
        logging.debug(f"Could not {action} process: {e}")
