"""Mean reversion: fade stretched moves, but only when there is no strong trend."""
import pandas as pd

from indicators import atr, bollinger, ema, rsi
from strategies.base import Strategy


class MeanReversion(Strategy):
    """
    Ranging filter: fast and slow EMA are within max_trend_atr * ATR of each other.
    CALL: close below the lower Bollinger band and RSI under rsi_low.
    PUT:  close above the upper band and RSI over rsi_high.
    Without the trend filter this strategy gets run over in trends.
    """

    name = "mean_reversion"

    def __init__(self, bb_n=20, bb_k=2.0, rsi_n=14, rsi_low=30, rsi_high=70,
                 fast=20, slow=50, max_trend_atr=1.0, hold_candles=3, strike_atr=0.7):
        super().__init__(hold_candles, strike_atr)
        self.bb_n, self.bb_k = bb_n, bb_k
        self.rsi_n, self.rsi_low, self.rsi_high = rsi_n, rsi_low, rsi_high
        self.fast, self.slow = fast, slow
        self.max_trend_atr = max_trend_atr

    @property
    def warmup(self) -> int:
        return max(self.slow, self.bb_n, self.rsi_n, self.atr_n) + 5

    def describe(self, direction: int) -> str:
        side = "below the lower band, oversold" if direction > 0 else "above the upper band, overbought"
        return f"price {side}, in a ranging market"

    def raw_signals(self, candles: pd.DataFrame) -> pd.Series:
        c = candles["close"]
        bb = bollinger(c, self.bb_n, self.bb_k)
        r = rsi(c, self.rsi_n)
        ranging = (ema(c, self.fast) - ema(c, self.slow)).abs() < self.max_trend_atr * atr(candles, self.atr_n)
        out = pd.Series(0, index=candles.index)
        out.loc[ranging & (c < bb["lower"]) & (r < self.rsi_low)] = 1
        out.loc[ranging & (c > bb["upper"]) & (r > self.rsi_high)] = -1
        return out