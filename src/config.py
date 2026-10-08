"""Central settings. Secrets come from the .env file, never from the code.

Run `python src/config.py` to check that your .env is found and filled in.
"""
import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

# --- Find and load the .env file -------------------------------------------
_HERE = Path(__file__).resolve().parent
_CANDIDATES = [_HERE / ".env", _HERE.parent / ".env", Path.cwd() / ".env"]
ENV_PATH = next((p for p in _CANDIDATES if p.is_file()), None)

if ENV_PATH is not None:
    # utf-8-sig survives the hidden BOM that Windows Notepad adds
    load_dotenv(ENV_PATH, override=True, encoding="utf-8-sig")


def _first(*names):
    """Return the first non-empty environment value among the given names."""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip().strip('"').strip("'")
    return None


# --- Credentials (several key names accepted, so your old names still work) --
_APP_ID_KEYS = ("DERIV_APP_ID", "APP_ID", "APPID")
_PAT_KEYS = ("DERIV_PAT", "PATAPI", "PAT", "DERIV_TOKEN")
_ACCOUNT_KEYS = ("DERIV_ACCOUNT_ID", "CLIENTid", "CLIENT_ID", "ACCOUNT_ID")

APP_ID = _first(*_APP_ID_KEYS)
PATAPI = _first(*_PAT_KEYS)
CLIENTid = _first(*_ACCOUNT_KEYS)

# --- Network -----------------------------------------------------------------
API_BASE = "https://api.derivws.com/trading/v1/options/accounts"
REQUEST_TIMEOUT = 15  # seconds

# --- Trading defaults (change freely; strategies will override later) --------
SYMBOL = "1HZ100V"        # Volatility 100 (1s) index
STAKE = 1.0               # stake per trade, in account currency
DURATION = 5              # contract length
DURATION_UNIT = "m"       # s, m, h, d or t
STRIKE_OFFSET_PCT = 0.1   # strike distance from spot, as a percent of spot

# --- Market data ---------------------------------------------------------------
CANDLE_SECONDS = 60       # candle size used by the live feed and downloads
HISTORY_COUNT = 500       # candles loaded at start-up so indicators are warm
DATA_DIR = _HERE.parent / "data"
LOG_DIR = _HERE.parent / "logs"

# --- Risk and staking -------------------------------------------------------------
STAKING_PLAN = "fixed_fraction"   # fixed_fraction, fixed, oscars_grind or anti_martingale
RISK_FRACTION = 0.01              # fixed_fraction: share of balance staked per trade (1%)
FIXED_STAKE = 1.0                 # fixed: stake per trade (the go-live gate wants <= MIN_STAKE)
MIN_STAKE = 1.0                   # smallest stake we will place
MAX_STAKE = 25.0                  # hard cap on any single stake
MAX_RISK_PCT_PER_TRADE = 2.0      # no stake may exceed this percent of balance
DAILY_LOSS_LIMIT_PCT = 3.0        # stop for the day after losing this percent of the day's start balance
MAX_CONSECUTIVE_LOSSES = 6        # stop for the day after this many losses in a row
MAX_TRADES_PER_DAY = 100
MAX_OPEN_POSITIONS = 1
MIN_SECONDS_BETWEEN_TRADES = 0
# --- Bot run settings ---------------------------------------------------------------
STRATEGY_NAME = "trend_pullback"  # trend_pullback, breakout or mean_reversion
DRY_RUN = True                    # True: log what would be traded but never buy. Set False for demo trades.
JOURNAL_PATH = LOG_DIR / "journal.db"
STATUS_EVERY_SECONDS = 300

MAX_MARKUP_PCT = 15.0             # placeholder: set it from the table that measure_markup.py prints
KILL_FILE = _HERE.parent / "STOP" # create a file with this name to stop the bot at once


if __name__ == "__main__":
    print(f".env file found at: {ENV_PATH}")
    if ENV_PATH is None:
        print("Looked in:")
        for p in _CANDIDATES:
            print(f"  {p}")
    else:
        names = [k for k in dotenv_values(ENV_PATH, encoding="utf-8-sig")]
        print(f"Key names inside it: {names}")
    print(f"App ID loaded:     {bool(APP_ID)}   (accepted names: {_APP_ID_KEYS})")
    print(f"Token loaded:      {bool(PATAPI)}   (accepted names: {_PAT_KEYS})")
    print(f"Account ID loaded: {bool(CLIENTid)}   (accepted names: {_ACCOUNT_KEYS})")