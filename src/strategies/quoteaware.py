"""Quote-aware selective direction: trade only when the proposal's EV beats its premium.

A new technical indicator will probably produce the same outcome as the existing
strategies, because the missing piece is not the signal but the price. So this module
prices the trade the other way round: it takes Deriv's actual call and put quotes as
given, asks what the option is worth under historical terminal moves, and only trades
when expected value exceeds the premium by a safety margin.

    expected payoff = mean(max(ST - K, 0)) * payoutperpoint
    call EV         = expected payoff - call premium

Both proposals must be requested at the same entry time, strike and expiry, because a
real straddle is `call premium + put premium` for simultaneously quoted contracts.
Averaging a directional strategy and its opposite does not reproduce that.

This is deliberately a no-trade-heavy strategy. If Deriv prices the contracts
efficiently it will choose no trades most of the time, which is preferable to forcing
thousands of negative-expectation positions.

Before live use, confirm that `payout_per_point` really is a per-point multiplier and
that one settled demo contract reproduces the API's reported profit exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Proposal:
    contract_type: str
    strike: float
    ask_price: float
    payout_per_point: float


@dataclass(frozen=True)
class Decision:
    side: "str | None"
    expected_value: float
    expected_roi: float
    reason: str


def historical_terminal_moves(
    close: pd.Series,
    entry_index: int,
    hold: int,
    lookback: int = 3000,
) -> np.ndarray:
    """Historical log moves over the same holding horizon, all known before entry.

    Samples are spaced `hold` candles apart so consecutive observations do not share
    the same future.
    """
    prices = close.to_numpy(dtype=float)
    first = max(0, entry_index - lookback)
    last_start = entry_index - hold

    if last_start <= first:
        return np.array([], dtype=float)

    starts = np.arange(first, last_start + 1, hold)
    valid = (
        np.isfinite(prices[starts])
        & np.isfinite(prices[starts + hold])
        & (prices[starts] > 0)
        & (prices[starts + hold] > 0)
    )
    starts = starts[valid]

    return np.log(prices[starts + hold] / prices[starts])


def proposal_ev(
    spot: float,
    terminal_log_moves: np.ndarray,
    proposal: Proposal,
) -> tuple[float, float]:
    """(expected value, expected ROI) of one proposal under the historical moves."""
    terminal_prices = spot * np.exp(terminal_log_moves)

    if proposal.contract_type == "VANILLALONGCALL":
        intrinsic = np.maximum(terminal_prices - proposal.strike, 0.0)
    elif proposal.contract_type == "VANILLALONGPUT":
        intrinsic = np.maximum(proposal.strike - terminal_prices, 0.0)
    else:
        raise ValueError(f"unsupported type: {proposal.contract_type}")

    expected_payout = float(np.mean(intrinsic) * proposal.payout_per_point)
    ev = expected_payout - proposal.ask_price
    roi = ev / proposal.ask_price
    return ev, roi


def choose_trade(
    close: pd.Series,
    entry_index: int,
    hold: int,
    call: Proposal,
    put: Proposal,
    minimum_samples: int = 150,
    minimum_expected_roi: float = 0.05,
    lookback: int = 3000,
) -> Decision:
    """Take the side whose proposal clears the EV threshold, or trade nothing."""
    moves = historical_terminal_moves(
        close=close,
        entry_index=entry_index,
        hold=hold,
        lookback=lookback,
    )

    if len(moves) < minimum_samples:
        return Decision(None, 0.0, 0.0, "insufficient history")

    spot = float(close.iloc[entry_index])
    call_ev, call_roi = proposal_ev(spot, moves, call)
    put_ev, put_roi = proposal_ev(spot, moves, put)

    if call_roi >= put_roi and call_roi >= minimum_expected_roi:
        return Decision("call", call_ev, call_roi, "call clears EV threshold")

    if put_roi > call_roi and put_roi >= minimum_expected_roi:
        return Decision("put", put_ev, put_roi, "put clears EV threshold")

    return Decision(
        None,
        max(call_ev, put_ev),
        max(call_roi, put_roi),
        "neither proposal clears EV threshold",
    )
