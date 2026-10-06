"""Replays a strategy over historical candles and prices each trade like Deriv would.

Rules (kept deliberately close to the live bot):
  * a signal fires on a closed candle; we enter at that candle's close price
  * the contract expires `hold_candles` candles later and pays by its real payoff
  * only one trade at a time; signals while a trade is open are ignored
  * we pay fair price x (1 + measured markup), see cost_model.py
  * for every trade we also compute what the OPPOSITE direction would have paid, so
    metrics.py can test whether the strategy's direction beats a coin flip
Entry is assumed at the signal candle's close. Real entries lag by a second or two,
so results are slightly optimistic.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.cost_model import LEVELS, CostModel
from indicators import atr
from pricing import SECONDS_PER_YEAR, bs_price, index_vol


@dataclass
class Trade:
    index: int
    epoch: int
    direction: int            # +1 CALL, -1 PUT
    level: float              # how far out of the money, in units of S*sigma*sqrt(T)
    spot: float
    strike: float
    expiry_price: float
    stake: float
    markup_pct: float
    fair_cost: float          # what this payoff was worth on average: stake / (1 + markup)
    payoff: float
    profit: float
    roi: float                # profit / stake
    roi_opposite: float       # same entry, opposite direction, per 1.0 staked


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    balances: list = field(default_factory=list)   # starts with the opening balance
    hold_candles: int = 0
    strategy: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])


def run_backtest(
    candles: pd.DataFrame,
    strategy,
    symbol: str,
    cost_model: CostModel,
    plan,
    start_balance: float = 1000.0,
    min_stake: float = 0.4,
    max_stake: float = 567.0,
    max_risk_pct: float = 2.0,
) -> BacktestResult:
    epoch = candles["epoch"].to_numpy()
    close = candles["close"].to_numpy(dtype=float)
    n = len(close)
    gran = int(np.median(np.diff(epoch)))
    sigma = index_vol(symbol)
    hold = strategy.hold_candles
    seconds = hold * gran
    T = seconds / SECONDS_PER_YEAR

    signals = strategy.signal_series(candles).to_numpy()
    atr_now = atr(candles, strategy.atr_n).to_numpy()

    balance = start_balance
    result = BacktestResult(balances=[balance], hold_candles=hold, strategy=strategy.name)
    open_until = 0

    for i in np.flatnonzero(signals):
        if i < open_until:
            continue
        j = i + hold
        if j >= n:
            break
        if epoch[j] - epoch[i] != seconds or math.isnan(atr_now[i]):
            continue  # a gap in the data, or ATR not ready

        direction = int(signals[i])
        kind, opposite = ("call", "put") if direction > 0 else ("put", "call")
        spot = close[i]
        unit = spot * sigma * math.sqrt(T)
        wanted = strategy.strike_atr * atr_now[i]
        level = min(LEVELS, key=lambda lv: abs(lv * unit - wanted))

        strike = spot + direction * level * unit
        strike_opp = spot - direction * level * unit
        fair = bs_price(kind, spot, strike, T, sigma)
        fair_opp = bs_price(opposite, spot, strike_opp, T, sigma)
        if fair < 1e-9 or fair_opp < 1e-9:
            continue

        markup = cost_model.markup_pct(seconds, level) / 100
        stake = min(max(plan.next_stake(balance), min_stake), max_stake, balance * max_risk_pct / 100)
        if stake < min_stake or stake > balance:
            break  # cannot afford another trade

        k = stake / (fair * (1 + markup))                      # payout per point
        payoff = k * max(direction * (close[j] - strike), 0.0)
        profit = payoff - stake

        k_opp = 1.0 / (fair_opp * (1 + markup))
        payoff_opp = k_opp * max(-direction * (close[j] - strike_opp), 0.0)

        result.trades.append(
            Trade(
                index=int(i), epoch=int(epoch[i]), direction=direction, level=level,
                spot=float(spot), strike=float(strike), expiry_price=float(close[j]),
                stake=float(stake), markup_pct=markup * 100, fair_cost=stake / (1 + markup),
                payoff=float(payoff), profit=float(profit), roi=float(profit / stake),
                roi_opposite=float(payoff_opp - 1.0),
            )
        )
        plan.record(profit)
        balance += profit
        result.balances.append(balance)
        open_until = j
    return result