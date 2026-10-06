"""Breakout: trade the first close outside the Bollinger Bands after a squeeze."""
import pandas as pd

from indicators import bollinger
from strategies.base import Strategy


class Breakout(Strategy):
    """
    Squeeze:  band width is in the lowest `squeeze_q` share of the last `lookback` candles.
    Trigger:  the candle after a squeeze closes above the upper band -> CALL,
              or below the lower band -> PUT.
    """

    name = "breakout"

    def __init__(self, bb_n=20, bb_k=2.0, lookback=100, squeeze_q=0.2,
                 hold_candles=3, strike_atr=0.7):
        super().__init__(hold_candles, strike_atr)
        self.bb_n, self.bb_k = bb_n, bb_k
        self.lookback, self.squeeze_q = lookback, squeeze_q

    @property
    def warmup(self) -> int:
        return self.bb_n + self.lookback + 2

    def describe(self, direction: int) -> str:
        side = "above the upper" if direction > 0 else "below the lower"
        return f"closed {side} Bollinger band right after a squeeze"

    def raw_signals(self, candles: pd.DataFrame) -> pd.Series:
        c = candles["close"]
        bb = bollinger(c, self.bb_n, self.bb_k)
        width = bb["width"]
        squeeze = width <= width.rolling(self.lookback).quantile(self.squeeze_q)
        was_squeezed = squeeze.shift(1, fill_value=False)
        out = pd.Series(0, index=candles.index)
        out.loc[was_squeezed & (c > bb["upper"])] = 1
        out.loc[was_squeezed & (c < bb["lower"])] = -1
        return out