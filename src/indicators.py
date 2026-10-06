"""Technical indicators. Pure functions on pandas objects, no trading logic.

`close` arguments are a pandas Series; `candles` arguments are a DataFrame with
open/high/low/close columns. Values before an indicator has enough data are NaN.
"""
import numpy as np
import pandas as pd


def sma(close: pd.Series, n: int) -> pd.Series:
    """Simple moving average."""
    return close.rolling(n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    """Exponential moving average (span n)."""
    return close.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing), 0 to 100."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.mask(avg_loss == 0, 100.0)                       # only gains
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)    # flat price
    return out.where(avg_gain.notna())


def true_range(candles: pd.DataFrame) -> pd.Series:
    prev_close = candles["close"].shift(1)
    parts = pd.concat(
        [
            candles["high"] - candles["low"],
            (candles["high"] - prev_close).abs(),
            (candles["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return parts.max(axis=1)


def atr(candles: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing), in price units."""
    return true_range(candles).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands. Columns: mid, upper, lower, width (upper - lower)."""
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "width": upper - lower})


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})