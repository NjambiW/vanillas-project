"""What Deriv charges, as a markup over the Black-Scholes fair price.

We have no historical Deriv quotes, so the backtester prices each trade as
    price paid = fair price x (1 + markup)
where the markup comes from real measurements (python src/measure_markup.py).

Strikes on Deriv sit at fixed distances from spot: 0, +/-0.68 and +/-1.29 times
S*sigma*sqrt(T) (this is what your measurements showed at every expiry). We call
that distance the "level": positive means out of the money, negative in the money.
"""
import csv
import glob
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

from pricing import index_vol, years

LEVELS = (-1.29, -0.68, 0.0, 0.68, 1.29)   # deep ITM ... ATM ... deep OTM

# Markup % over fair price, measured on Deriv 1HZ100V on 2026-10-06 (average of the
# CALL and PUT rows). Columns follow LEVELS. Seconds -> five markups.
MEASURED = {
    60: [10.1, 14.0, 21.0, 31.0, 41.9],       # 1 minute
    300: [4.8, 7.0, 11.1, 17.3, 24.9],        # 5 minutes
    900: [2.9, 4.6, 7.6, 12.7, 19.2],         # 15 minutes
    3600: [1.7, 2.9, 5.4, 9.6, 15.4],         # 1 hour
    14400: [1.1, 2.0, 4.2, 8.0, 13.4],        # 4 hours
}

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


class CostModel:
    def __init__(self, table: Optional[dict] = None):
        self.table = dict(table or MEASURED)
        self._seconds = sorted(self.table)

    def markup_pct(self, seconds: float, level: float) -> float:
        """Markup in percent, interpolated across level and (log) time, clamped at the edges."""
        per_duration = [float(np.interp(level, LEVELS, self.table[s])) for s in self._seconds]
        if len(self._seconds) == 1:
            return per_duration[0]
        return float(np.interp(math.log(seconds), np.log(self._seconds), per_duration))


def _parse_duration(text: str) -> int:
    match = re.fullmatch(r"(\d+)([smhd])", text.strip())
    if not match:
        raise ValueError(f"bad duration {text!r}")
    return int(match.group(1)) * _UNIT_SECONDS[match.group(2)]


def load_markup_csv(path, symbol: str = "1HZ100V") -> dict:
    """Build a markup table from a CSV written by measure_markup.py.

    Only durations with all five strike levels measured are used.
    """
    sigma = index_vol(symbol)
    cells: dict = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                seconds = _parse_duration(row["duration"])
                markup = float(row["markup_pct"])
                spot, strike = float(row["spot"]), float(row["strike"])
            except (KeyError, ValueError):
                continue
            if markup != markup:  # NaN
                continue
            unit = spot * sigma * math.sqrt(years(seconds, "s"))
            offset = strike - spot
            level = offset / unit if row["type"].endswith("CALL") else -offset / unit
            idx = min(range(len(LEVELS)), key=lambda i: abs(LEVELS[i] - level))
            cells.setdefault(seconds, {}).setdefault(idx, []).append(markup)
    table = {}
    for seconds, by_idx in cells.items():
        if len(by_idx) == len(LEVELS):
            table[seconds] = [round(sum(by_idx[i]) / len(by_idx[i]), 2) for i in range(len(LEVELS))]
    return table


def latest_markup_csv(log_dir) -> Optional[Path]:
    files = sorted(glob.glob(str(Path(log_dir) / "markup_*.csv")))
    return Path(files[-1]) if files else None