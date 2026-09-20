"""Playback control routes for skip, pause, volume, transpose, and vocals."""

import flask_babel
from flask import abort, redirect, request, url_for
from flask_smorest import Blueprint

from pikaraoke.lib.current_app import broadcast_event, get_karaoke_instance

_ = flask_babel.gettext


controller_bp = Blueprint("controller", __name__)


@controller_bp.route("/skip", methods=["POST"])
def skip():
    """Skip the currently playing song."""
    k = get_karaoke_instance()
    broadcast_event("skip", "user command")
    k.playback_controller.skip()
    return redirect(url_for("home.home"))


@controller_bp.route("/pause", methods=["POST"])
def pause():
    """Toggle pause/resume playback."""
    k = get_karaoke_instance()
    if k.playback_controller.is_paused:
        broadcast_event("play")
    else:
        broadcast_event("pause")
    k.playback_controller.pause()
    return redirect(url_for("home.home"))


@controller_bp.route("/transpose/<semitones>", methods=["POST"])
def transpose(semitones):
    """Transpose (pitch shift) the current song.

    If the requested pitch was already pre-rendered, switch to it directly
    (no restart). Otherwise fall back to restarting the song at the new
    pitch, same as before pre-rendering existed.
    """
    k = get_karaoke_instance()
    s = int(semitones)
    if k.playback_controller.can_fast_switch(s):
        k.fast_transpose(s)
    else:
        broadcast_event("skip", "transpose current")
        k.transpose_current(s)
    return redirect(url_for("home.home"))


@controller_bp.route("/vocals/<state>", methods=["POST"])
def vocals(state):
    """Switch the current song's vocals on or off.

    Takes an explicit state rather than toggling, so two remotes pressing the
    button at once can't cancel each other out.
    """
    if state not in ("on", "off"):
        abort(400)
    k = get_karaoke_instance()
    k.set_vocals(state == "on")
    return redirect(url_for("home.home"))


@controller_bp.route("/restart", methods=["POST"])
def restart():
    """Restart the current song from the beginning."""
    k = get_karaoke_instance()
    broadcast_event("restart")
    k.restart()
    return redirect(url_for("home.home"))


@controller_bp.route("/volume/<volume>", methods=["POST"])
def volume(volume):
    """Set the playback volume."""
    k = get_karaoke_instance()
    broadcast_event("volume", volume)
    k.volume_change(float(volume))
    return redirect(url_for("home.home"))


@controller_bp.route("/vol_up", methods=["POST"])
def vol_up():
    """Increase volume by 10%."""
    k = get_karaoke_instance()
    broadcast_event("volume", "up")
    k.vol_up()
    return redirect(url_for("home.home"))


@controller_bp.route("/vol_down", methods=["POST"])
def vol_down():
    """Decrease volume by 10%."""
    k = get_karaoke_instance()
    broadcast_event("volume", "down")
    k.vol_down()
    return redirect(url_for("home.home"))
