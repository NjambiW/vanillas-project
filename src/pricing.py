"""Option pricing: Black-Scholes, Greeks, implied volatility.

All prices are in index points (price units). Multiply by payout-per-point to get
money. Interest rate defaults to zero, which is right for short-dated synthetics.
"""
import math
import re

SECONDS_PER_YEAR = 365 * 24 * 3600
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def years(duration: int, unit: str) -> float:
    """Contract length in years. Ticks are not supported (their length varies)."""
    if unit not in _UNIT_SECONDS:
        raise ValueError(f"unsupported duration unit: {unit!r}")
    return duration * _UNIT_SECONDS[unit] / SECONDS_PER_YEAR


def index_vol(symbol: str) -> float:
    """Annual volatility of a Deriv Volatility index, from its name.

    1HZ100V and R_100 both mean 100% (returned as 1.0); 1HZ10V and R_10 mean 10%.
    """
    match = re.fullmatch(r"(?:1HZ|R_)(\d+)V?", symbol)
    if not match:
        raise ValueError(f"cannot read the volatility from symbol {symbol!r}")
    return int(match.group(1)) / 100


def _d1_d2(S: float, K: float, T: float, sigma: float, r: float):
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_put(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_price(kind: str, S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """kind is 'call' or 'put'."""
    if kind == "call":
        return bs_call(S, K, T, sigma, r)
    if kind == "put":
        return bs_put(S, K, T, sigma, r)
    raise ValueError("kind must be 'call' or 'put'")


def greeks(kind: str, S: float, K: float, T: float, sigma: float, r: float = 0.0) -> dict:
    """delta, gamma, vega (per 1.00 of vol), theta (per year, in points)."""
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    pdf = norm_pdf(d1)
    delta = norm_cdf(d1) if kind == "call" else norm_cdf(d1) - 1
    gamma = pdf / (S * sigma * math.sqrt(T))
    vega = S * pdf * math.sqrt(T)
    decay = -S * pdf * sigma / (2 * math.sqrt(T))
    discounted_strike = K * math.exp(-r * T)
    if kind == "call":
        theta = decay - r * discounted_strike * norm_cdf(d2)
    else:
        theta = decay + r * discounted_strike * norm_cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def implied_vol(
    price: float,
    kind: str,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    lo: float = 1e-4,
    hi: float = 20.0,
) -> float:
    """Volatility that makes Black-Scholes equal `price`. NaN if impossible."""
    if T <= 0:
        return float("nan")
    low_price = bs_price(kind, S, K, T, lo, r)
    high_price = bs_price(kind, S, K, T, hi, r)
    if not (low_price <= price <= high_price):
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs_price(kind, S, K, T, mid, r) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


def expected_move(S: float, sigma: float, T: float) -> float:
    """One-standard-deviation price move over T years: S * sigma * sqrt(T)."""
    return S * sigma * math.sqrt(T)