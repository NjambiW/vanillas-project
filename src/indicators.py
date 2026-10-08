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


def adx(candles: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's Average Directional Index (0-100), smoothed over n periods.

    ADX quantifies trend strength. Readings below ~20 often imply a ranging market,
    above ~25 a trending market. Values before sufficient data are NaN.
    """
    high = candles["high"]
    low = candles["low"]
    close = candles["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = true_range(candles)

    trn = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdm = plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    mdm = minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    plus_di = pdm / trn.replace(0, np.nan) * 100.0
    minus_di = mdm / trn.replace(0, np.nan) * 100.0

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100.0

    adx_s = dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return adx_s


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})