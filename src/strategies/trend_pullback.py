"""Trend pullback: buy the bounce back into a confirmed trend."""
import numpy as np
import pandas as pd

from indicators import atr, ema
from strategies.base import Strategy


class TrendPullback(Strategy):
    """
    Uptrend:  fast EMA above slow EMA, slow EMA rising, gap wider than min_trend_atr * ATR.
    Trigger:  price had closed below the fast EMA and now closes back above it -> CALL.
    Downtrend is the mirror image -> PUT.
    Cooldown: after a signal, further signals are suppressed for `cooldown_candles` candles
              (defaults to hold_candles, i.e. no new signal while the trade would still be open).
    The default numbers are starting points to be tuned by backtesting, not facts.
    """

    name = "trend_pullback"

    def __init__(self, fast=20, slow=50, slope_lookback=5, min_trend_atr=0.2,
                 hold_candles=5, strike_atr=0.7, cooldown_candles=None):
        super().__init__(hold_candles, strike_atr)
        self.fast, self.slow = fast, slow
        self.slope_lookback = slope_lookback
        self.min_trend_atr = min_trend_atr
        self.cooldown_candles = hold_candles if cooldown_candles is None else cooldown_candles

    @property
    def warmup(self) -> int:
        return max(self.slow, self.atr_n) + self.slope_lookback + 5

    def describe(self, direction: int) -> str:
        side = "up" if direction > 0 else "down"
        return f"price reclaimed EMA{self.fast} inside a confirmed {side}trend"

    def raw_signals(self, candles: pd.DataFrame) -> pd.Series:
        c = candles["close"]
        fast, slow = ema(c, self.fast), ema(c, self.slow)
        a = atr(candles, self.atr_n)
        strong = (fast - slow).abs() > self.min_trend_atr * a
        uptrend = (fast > slow) & (slow > slow.shift(self.slope_lookback)) & strong
        downtrend = (fast < slow) & (slow < slow.shift(self.slope_lookback)) & strong
        reclaim_up = (c.shift(1) < fast.shift(1)) & (c > fast)
        reclaim_down = (c.shift(1) > fast.shift(1)) & (c < fast)
        out = pd.Series(0, index=candles.index)
        out.loc[uptrend & reclaim_up] = 1
        out.loc[downtrend & reclaim_down] = -1
        return self._apply_cooldown(out)

    def _apply_cooldown(self, signals: pd.Series) -> pd.Series:
        if self.cooldown_candles <= 0:
            return signals
        values = signals.to_numpy().copy()
        last = -10 ** 9
        for pos in np.flatnonzero(values):
            if pos - last < self.cooldown_candles:
                values[pos] = 0
            else:
                last = pos
        return pd.Series(values, index=signals.index)