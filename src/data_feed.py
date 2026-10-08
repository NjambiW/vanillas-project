"""Market data: tick-to-candle building, a live candle feed, and CSV helpers."""
from __future__ import annotations

import logging
import os
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import pandas as pd

if TYPE_CHECKING:  # avoids importing the websocket layer just to use the helpers
    from deriv_client import DerivClient

log = logging.getLogger("data_feed")

CANDLE_COLUMNS = ["epoch", "open", "high", "low", "close"]

# Candle sizes the history endpoint accepts in practice. Deriv's docs say any
# integer is allowed, but the server rejected 10 seconds, so we only ask for
# these and build any other size ourselves from ticks.
STANDARD_GRANULARITIES = {60, 120, 180, 300, 600, 900, 1800, 3600, 7200, 14400, 28800, 86400}
MAX_TICKS_PER_REQUEST = 5000


# ----------------------------------------------------------------- helpers
def candles_to_frame(candles: list[dict]) -> pd.DataFrame:
    """List of candle dicts to a DataFrame indexed by UTC time, oldest first."""
    if not candles:
        df = pd.DataFrame(columns=CANDLE_COLUMNS)
    else:
        df = pd.DataFrame(candles)[CANDLE_COLUMNS]
    df = df.astype({c: float for c in CANDLE_COLUMNS})
    df["epoch"] = df["epoch"].astype("int64")
    df = df.drop_duplicates("epoch").sort_values("epoch").reset_index(drop=True)
    df.index = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df.index.name = "time"
    return df


def candles_from_ticks(ticks: list[dict], granularity: int) -> list[dict]:
    """Aggregate [{epoch, price}, ...] into candle dicts of any size."""
    if not ticks:
        return []
    df = pd.DataFrame(ticks).sort_values("epoch", kind="stable")
    df["bucket"] = df["epoch"] - df["epoch"] % granularity
    grouped = df.groupby("bucket")["price"]
    out = pd.DataFrame(
        {
            "epoch": grouped.first().index.astype("int64"),
            "open": grouped.first().values,
            "high": grouped.max().values,
            "low": grouped.min().values,
            "close": grouped.last().values,
        }
    )
    return out.to_dict("records")


def save_candles_csv(df: pd.DataFrame, path: Path) -> None:
    """Write the candles out atomically so a crash can never leave a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df[CANDLE_COLUMNS].to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_candles_csv(path: Path) -> pd.DataFrame:
    return candles_to_frame(pd.read_csv(path).to_dict("records"))


# ----------------------------------------------------------------- builder
class CandleBuilder:
    """Turns a stream of ticks into fixed-size candles."""

    def __init__(self, granularity: int, max_candles: int = 2000):
        self.granularity = granularity
        self._closed: deque[dict] = deque(maxlen=max_candles)
        self._current: Optional[dict] = None

    def _bucket(self, epoch: int) -> int:
        return epoch - epoch % self.granularity

    def seed(self, candles: list[dict]) -> None:
        """Load history. The newest candle is treated as the one still forming."""
        rows = [
            {k: float(c[k]) if k != "epoch" else int(c[k]) for k in CANDLE_COLUMNS}
            for c in sorted(candles, key=lambda c: c["epoch"])
        ]
        self._closed.clear()
        self._current = None
        if not rows:
            return
        self._closed.extend(rows[:-1])
        self._current = rows[-1]

    def on_tick(self, epoch: int, price: float) -> Optional[dict]:
        """Feed one tick. Returns the candle that just closed, or None."""
        bucket = self._bucket(epoch)
        cur = self._current
        if cur is None:
            self._current = self._new_candle(bucket, price)
            return None
        if bucket == cur["epoch"]:
            cur["high"] = max(cur["high"], price)
            cur["low"] = min(cur["low"], price)
            cur["close"] = price
            return None
        if bucket < cur["epoch"]:
            return None  # late or duplicate tick, ignore
        closed = dict(cur)
        self._closed.append(closed)
        self._current = self._new_candle(bucket, price)
        return closed

    @staticmethod
    def _new_candle(bucket: int, price: float) -> dict:
        return {"epoch": bucket, "open": price, "high": price, "low": price, "close": price}

    def frame(self, include_forming: bool = False) -> pd.DataFrame:
        rows = list(self._closed)
        if include_forming and self._current is not None:
            rows.append(dict(self._current))
        return candles_to_frame(rows)


# -------------------------------------------------------------------- feed
class CandleFeed:
    """Live candles for one symbol. Calls on_candle(frame) each time a candle closes."""

    def __init__(
        self,
        client: "DerivClient",
        symbol: str,
        granularity: int = 60,
        history_count: int = 500,
        on_candle: Optional[Callable[[pd.DataFrame], None]] = None,
    ):
        self.client = client
        self.symbol = symbol
        self.granularity = granularity
        self.history_count = history_count
        self.on_candle = on_candle
        self.builder = CandleBuilder(granularity, max_candles=max(history_count * 2, 1000))
        self.last_price: Optional[float] = None
        self._subscription_id: Optional[str] = None

    async def start(self) -> None:
        if self.granularity in STANDARD_GRANULARITIES:
            history = await self.client.get_candles(
                self.symbol, self.granularity, self.history_count
            )
        else:
            # Unusual candle size: build the history from ticks instead.
            wanted = min(self.history_count * self.granularity, MAX_TICKS_PER_REQUEST)
            ticks = await self.client.get_tick_history(self.symbol, wanted)
            history = candles_from_ticks(ticks, self.granularity)[-self.history_count :]
        self.builder.seed(history)
        log.info("seeded %d historical candles", len(history))
        first = await self.client.subscribe_ticks(self.symbol, self._on_tick_message)
        self._subscription_id = (first.get("subscription") or {}).get("id")

    async def stop(self) -> None:
        if self._subscription_id:
            await self.client.forget(self._subscription_id)
            self._subscription_id = None

    def _on_tick_message(self, msg: dict) -> None:
        tick = msg.get("tick")
        if not tick:
            return
        price = float(tick["quote"])
        self.last_price = price
        closed = self.builder.on_tick(int(tick["epoch"]), price)
        if closed is not None and self.on_candle is not None:
            self.on_candle(self.builder.frame())

    def frame(self, include_forming: bool = False) -> pd.DataFrame:
        return self.builder.frame(include_forming)