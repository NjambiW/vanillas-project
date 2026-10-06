from unittest.mock import MagicMock

import pytest
import requests

import authenticator
from authenticator import AuthError, get_websocket_url


@pytest.fixture(autouse=True)
def fake_credentials(monkeypatch):
    monkeypatch.setattr(authenticator, "APP_ID", "app123")
    monkeypatch.setattr(authenticator, "PATAPI", "pat_secret")
    monkeypatch.setattr(authenticator, "CLIENTid", "acct456")


def make_response(status=200, payload=None, text=""):
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.json.return_value = payload
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(str(status))
    return resp


def test_returns_websocket_url(monkeypatch):
    resp = make_response(payload={"data": {"url": "wss://example.test/ws?otp=abc"}})
    monkeypatch.setattr(authenticator.requests, "post", MagicMock(return_value=resp))
    assert get_websocket_url() == "wss://example.test/ws?otp=abc"


def test_sends_correct_request(monkeypatch):
    resp = make_response(payload={"data": {"url": "wss://x"}})
    post = MagicMock(return_value=resp)
    monkeypatch.setattr(authenticator.requests, "post", post)

    get_websocket_url()

    kwargs = post.call_args.kwargs
    assert kwargs["url"].endswith("/acct456/otp")
    assert kwargs["headers"]["Deriv-App-ID"] == "app123"
    assert kwargs["headers"]["Authorization"] == "Bearer pat_secret"
    assert kwargs["timeout"] > 0


def test_missing_credentials_raises(monkeypatch):
    monkeypatch.setattr(authenticator, "PATAPI", None)
    with pytest.raises(AuthError, match="DERIV_PAT"):
        get_websocket_url()


def test_http_error_raises_autherror(monkeypatch):
    resp = make_response(status=401, text="invalid token")
    monkeypatch.setattr(authenticator.requests, "post", MagicMock(return_value=resp))
    with pytest.raises(AuthError, match="401"):
        get_websocket_url()


def test_bad_response_shape_raises(monkeypatch):
    resp = make_response(payload={"unexpected": True})
    monkeypatch.setattr(authenticator.requests, "post", MagicMock(return_value=resp))
    with pytest.raises(AuthError, match="Unexpected response"):
        get_websocket_url()


def test_url_is_never_printed(monkeypatch, capsys):
    resp = make_response(payload={"data": {"url": "wss://secret.test/ws?otp=abc"}})
    monkeypatch.setattr(authenticator.requests, "post", MagicMock(return_value=resp))
    get_websocket_url()
    out = capsys.readouterr()
    assert "secret.test" not in out.out + out.err