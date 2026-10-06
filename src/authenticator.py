"""Gets the authenticated WebSocket URL from the Deriv API."""
import requests

from config import API_BASE, APP_ID, CLIENTid, PATAPI, REQUEST_TIMEOUT


class AuthError(Exception):
    """Raised when we cannot obtain a WebSocket URL."""


def get_websocket_url() -> str:
    """Request a one-time-password URL and return the WebSocket URL.

    The returned URL carries a credential, so it is never printed or logged here.
    Call this again for every new connection or reconnect.
    """
    missing = [
        name
        for name, value in (
            ("DERIV_APP_ID", APP_ID),
            ("DERIV_PAT", PATAPI),
            ("DERIV_ACCOUNT_ID", CLIENTid),
        )
        if not value
    ]
    if missing:
        raise AuthError(f"Missing values in .env: {', '.join(missing)}")

    headers = {
        "Deriv-App-ID": APP_ID,
        "Authorization": f"Bearer {PATAPI}",
    }
    url = f"{API_BASE}/{CLIENTid}/otp"

    response = requests.post(url=url, headers=headers, timeout=REQUEST_TIMEOUT)

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise AuthError(
            f"Deriv rejected the request (HTTP {response.status_code}): {response.text}"
        ) from exc

    try:
        return response.json()["data"]["url"]
    except (ValueError, KeyError, TypeError) as exc:
        raise AuthError("Unexpected response shape from Deriv (no data.url).") from exc


if __name__ == "__main__":
    ws_url = get_websocket_url()
    print(f"Authenticated OK. Got a WebSocket URL ({len(ws_url)} characters).")