"""Compare Deriv's vanilla price with the Black-Scholes fair price.

Pure logic, no network. The script that talks to Deriv is measure_markup.py.
"""
import math

from pricing import bs_price, implied_vol, index_vol, years

# On Deriv vanillas the payout per point is reported as "display_number_of_contracts".
PAYOUT_PER_POINT_KEYS = ("payout_per_point", "payoutPerPoint", "display_number_of_contracts")
ASK_PRICE_KEYS = ("ask_price", "display_value")
SPOT_KEYS = ("spot", "entry_spot")


class MarkupError(Exception):
    """The proposal did not contain something we need."""


def find_field(obj, names):
    """Find the first non-empty value for any of `names`, searching nested data."""
    if isinstance(obj, dict):
        for name in names:
            value = obj.get(name)
            if value not in (None, ""):
                return value
        for value in obj.values():
            found = find_field(value, names)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_field(value, names)
            if found is not None:
                return found
    return None


def _absolute_strike(proposal: dict):
    """The exact strike Deriv will use, if the proposal reports it (e.g. '1226.39')."""
    details = proposal.get("contract_details")
    if isinstance(details, dict):
        raw = str(details.get("barrier") or "")
        if raw and not raw.startswith(("+", "-")):
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def markup_row(
    contract_type: str,
    proposal: dict,
    fallback_spot: float,
    barrier: str,
    duration: int,
    unit: str,
    symbol: str,
) -> dict:
    """Turn one proposal into a row: price paid, fair price, markup, implied vol."""
    kind = "call" if contract_type.endswith("CALL") else "put"

    ask = find_field(proposal, ASK_PRICE_KEYS)
    k = find_field(proposal, PAYOUT_PER_POINT_KEYS)
    if ask is None or k is None:
        raise MarkupError(
            "could not find ask_price / payout_per_point in the proposal. "
            f"Keys seen: {sorted(proposal)}"
        )
    ask, k = float(ask), float(k)
    spot = float(find_field(proposal, SPOT_KEYS) or fallback_spot)

    strike = _absolute_strike(proposal)
    if strike is None:
        # "+2.20" is an offset from the entry spot; "1240.00" is the strike itself
        strike = spot + float(barrier) if str(barrier).strip().startswith(("+", "-")) else float(barrier)
    T = years(duration, unit)
    sigma = index_vol(symbol)

    fair_points = bs_price(kind, spot, strike, T, sigma)
    fair_stake = k * fair_points
    markup_pct = (ask / fair_stake - 1) * 100 if fair_stake > 1e-12 else math.nan
    iv = implied_vol(ask / k, kind, spot, strike, T) if k > 0 else math.nan

    return {
        "type": contract_type,
        "duration": f"{duration}{unit}",
        "barrier": barrier,
        "spot": spot,
        "strike": round(strike, 4),
        "ask_price": ask,
        "payout_per_point": k,
        "fair_price": round(fair_stake, 6),
        "markup_pct": round(markup_pct, 2) if not math.isnan(markup_pct) else math.nan,
        "implied_vol": round(iv, 4) if not math.isnan(iv) else math.nan,
        "model_vol": sigma,
    }