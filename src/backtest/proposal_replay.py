"""Replays trades priced from the proposals Deriv actually offered.

The estimated-pricing backtest answers "would this signal have beaten a Black-Scholes
fair value plus a measured markup?". That diagnoses our pricing model; it cannot
establish tradability. Deriv's premium also depends on entry spot, strike, duration,
payout-per-point, volatility and recent path, proposal time, contract type, rounding,
quote changes and stake limits -- and payout per point varies with the strike/spot
relationship, so one percentage markup table cannot reproduce that surface.

So the executor appends every quote it sees, bought or rejected, to an append-only
log (see proposal_log.py) and this module settles those quotes against the candle
history using only the observed numbers:

    payoff = payoutperpoint * max(+- (ST - K), 0)
    profit = payoff - askprice
    roi    = profit / askprice

No model price appears anywhere in that calculation, which is the point.

Approximations, all inherited from having only 1-minute candles:
  * the entry is snapped to the open of the candle containing `epoch`;
  * settlement uses the close of the candle `duration` later, not Deriv's exact
    tick at expiry;
  * the strike is rebuilt from the logged barrier and spot rather than read from
    the contract's resolved absolute strike.
Any quote that lands on a gap in the candle history is dropped rather than guessed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.walkforward import blocksize_for, summarize_fold_trades

UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
NEEDED = ("epoch", "contracttype", "duration", "durationunit", "barrier",
          "spot", "askprice", "payoutperpoint")


def load_proposals(path: "str | Path", symbol: str | None = None) -> pd.DataFrame:
    """Read a proposal log, returning an empty frame if there is nothing usable."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        # Read every column as text: pandas would otherwise infer `barrier` as a
        # number and silently turn a relative "+0.00" into 0.0, losing the sign
        # that says it is relative to spot.
        df = pd.read_csv(path, dtype=str)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()
    if df.empty or not set(NEEDED).issubset(df.columns):
        return pd.DataFrame()
    if symbol is not None and "symbol" in df.columns:
        df = df[df["symbol"].astype(str) == str(symbol)]

    df = df.copy()
    df["barrier"] = df["barrier"].astype(str).str.strip()
    for col in ("epoch", "duration", "spot", "askprice", "payoutperpoint"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    unit = df["durationunit"].astype(str).str.strip().str[:1].str.lower()
    df["duration_seconds"] = df["duration"] * unit.map(UNIT_SECONDS)
    df = df.dropna(subset=["epoch", "duration_seconds", "spot", "askprice", "payoutperpoint"])
    return df.reset_index(drop=True)


def replay(proposals: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Settle every logged quote against the candle history."""
    if proposals.empty:
        return pd.DataFrame()

    epoch = candles["epoch"].to_numpy()
    close = candles["close"].to_numpy(dtype=float)
    gran = int(np.median(np.diff(epoch))) if len(epoch) > 1 else 0
    if gran <= 0:
        return pd.DataFrame()

    rows = []
    for r in proposals.itertuples(index=False):
        entry = int(np.searchsorted(epoch, r.epoch, side="right")) - 1
        if entry < 0:
            continue
        steps = int(round(r.duration_seconds / gran))
        if steps < 1:
            continue
        expiry = entry + steps
        if expiry >= len(epoch):
            continue
        if epoch[expiry] - epoch[entry] != steps * gran:
            continue  # a gap in the candles: settle nothing by guesswork

        spot, barrier = float(r.spot), str(r.barrier).strip()
        ask, per_point = float(r.askprice), float(r.payoutperpoint)
        if not (spot > 0 and ask > 0 and per_point > 0):
            continue
        try:
            value = float(barrier)
        except ValueError:
            continue
        strike = spot + value if barrier[:1] in ("+", "-") else value

        final = float(close[expiry])
        contract = str(r.contracttype)
        if contract.endswith("CALL"):
            intrinsic = max(final - strike, 0.0)
        elif contract.endswith("PUT"):
            intrinsic = max(strike - final, 0.0)
        else:
            continue

        payoff = per_point * intrinsic
        profit = payoff - ask
        rows.append({
            "epoch": int(r.epoch),
            "contracttype": contract,
            "barrier": barrier,
            "spot": spot,
            "strike": strike,
            "expiry_price": final,
            "askprice": ask,
            "payoutperpoint": per_point,
            "payoff": payoff,
            "profit": profit,
            "roi": profit / ask,
        })
    return pd.DataFrame(rows)


def replay_report(path: "str | Path", candles: pd.DataFrame, symbol: str | None = None) -> dict:
    """Summary of a proposal log's replay, or `None` if there is no data to replay.

    Returns None when the log is absent or unusable -- a missing log is not a failed
    result, it means the check has not been made yet.
    """
    proposals = load_proposals(path, symbol)
    if proposals.empty:
        return None

    settled = replay(proposals, candles)
    report = {
        "path": str(path),
        "quotes": len(proposals),
        "settled": len(settled),
        "df": settled,
        "mean": float("nan"), "cilow": float("nan"), "cihigh": float("nan"),
        "profitable": False, "block": 1,
    }
    if settled.empty:
        return report

    epoch = candles["epoch"].to_numpy()
    gran = int(np.median(np.diff(epoch)))
    hold = max(1, int(round(float(proposals["duration_seconds"].median()) / max(1, gran))))
    block = blocksize_for(hold, len(settled))
    ci = summarize_fold_trades(settled, returncolumn="roi", blocksize=block)
    report.update({"mean": ci["mean"], "cilow": ci["cilow"], "cihigh": ci["cihigh"],
                   "block": block, "profitable": ci["profitable"]})
    return report
