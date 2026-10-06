"""Strategy interface shared by the backtester and the live bot.

Rules every strategy follows:
  * It only looks at CLOSED candles, and the value at row i uses data up to row i.
    (There is a test that fails if a strategy peeks at the future.)
  * It says what it wants (CALL or PUT) and how far away and how long, in terms of
    ATR and candle counts. Turning that into an actual Deriv strike and expiry is
    the executor's job, because Deriv only offers fixed strikes.
  * It does not manage open trades, stakes or limits. That belongs to risk.py.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from indicators import atr


@dataclass(frozen=True)
class Signal:
    direction: str          # "CALL" or "PUT"
    reason: str             # short human-readable explanation, goes in the journal
    atr: float              # ATR at signal time, in index points
    strike_atr: float       # wanted strike distance from spot, in ATRs
    hold_candles: int       # how many candles the idea should play out over

    @property
    def contract_type(self) -> str:
        return "VANILLALONGCALL" if self.direction == "CALL" else "VANILLALONGPUT"

    @property
    def strike_distance(self) -> float:
        """Wanted strike distance in index points."""
        return self.strike_atr * self.atr


class Strategy(ABC):
    name = "base"
    atr_n = 14

    def __init__(self, hold_candles: int = 5, strike_atr: float = 0.7):
        self.hold_candles = hold_candles
        self.strike_atr = strike_atr

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Minimum number of candles before signals are trusted."""

    @abstractmethod
    def raw_signals(self, candles: pd.DataFrame) -> pd.Series:
        """+1 = CALL, -1 = PUT, 0 = nothing, one value per candle, causal."""

    @abstractmethod
    def describe(self, direction: int) -> str:
        """Reason text for a +1 / -1 signal."""

    def signal_series(self, candles: pd.DataFrame) -> pd.Series:
        """Signals for every candle. The backtester uses this directly."""
        out = self.raw_signals(candles).fillna(0).astype(int)
        out.iloc[: self.warmup] = 0
        return out

    def evaluate(self, candles: pd.DataFrame) -> Optional[Signal]:
        """Live use: look at the newest closed candle and return a Signal or None."""
        if len(candles) < self.warmup:
            return None
        value = int(self.signal_series(candles).iloc[-1])
        if value == 0:
            return None
        atr_now = float(atr(candles, self.atr_n).iloc[-1])
        return Signal(
            direction="CALL" if value > 0 else "PUT",
            reason=self.describe(value),
            atr=atr_now,
            strike_atr=self.strike_atr,
            hold_candles=self.hold_candles,
        )