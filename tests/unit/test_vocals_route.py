"""The /vocals route, which switches the playing song's vocals on or off."""

from unittest.mock import MagicMock, patch

import pytest
import werkzeug
from flask import Blueprint, Flask

if not hasattr(werkzeug, "__version__"):
    werkzeug.__version__ = "3.0.0"

from pikaraoke.routes.controller import controller_bp


@pytest.fixture
def karaoke():
    return MagicMock()


@pytest.fixture
def client(karaoke):
    app = Flask(__name__)
    app.register_blueprint(controller_bp)
    # The route redirects home; only the endpoint needs to exist.
    home = Blueprint("home", __name__)
    home.add_url_rule("/", "home", lambda: "home")
    app.register_blueprint(home)
    with patch("pikaraoke.routes.controller.get_karaoke_instance", return_value=karaoke):
        yield app.test_client()


def test_off_switches_vocals_off(client, karaoke):
    client.post("/vocals/off")

    karaoke.set_vocals.assert_called_once_with(False)


def test_on_switches_vocals_on(client, karaoke):
    client.post("/vocals/on")

    karaoke.set_vocals.assert_called_once_with(True)


@pytest.mark.parametrize("state", ["toggle", "1", "OFF", ""])
def test_anything_but_on_or_off_is_rejected(client, karaoke, state):
    """An explicit state, so two remotes pressing at once can't cancel each other out."""
    response = client.post(f"/vocals/{state}")

    assert response.status_code in (400, 404)
    karaoke.set_vocals.assert_not_called()
