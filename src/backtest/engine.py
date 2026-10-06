"""Replays a strategy over historical candles and prices each trade like Deriv would.

Rules (kept deliberately close to the live bot):
  * a signal fires on a closed candle; we enter at that candle's close price
  * the contract expires `hold_candles` candles later and pays by its real payoff
  * only one trade at a time; signals while a trade is open are ignored
  * we pay fair price x (1 + measured markup), see cost_model.py
  * LIVE RISK RULES are applied (risk.py's RiskConfig): daily loss limit, max consecutive
    losses, max trades per day, cooldown between trades, stake sizing (min/max stake and the
    percent-of-balance cap). Automatic halts last until the next UTC day, like live.
  * the PREMIUM GATE is applied: among the `max_candidates` strike levels closest to the
    wanted strike we take the one with the lowest markup, and skip the signal if even that
    exceeds the limit (the same rule the live executor follows)
  * for every trade we also compute what the OPPOSITE direction would have paid, so
    metrics.py can test whether the strategy's direction beats a coin flip
Entry is assumed at the signal candle's close. Real entries lag by a second or two,
so results are slightly optimistic.

Options:
  risk_cfg        None  -> use RiskConfig.from_settings() (your config.py) if it can be loaded,
                           otherwise fall back to the min_stake/max_stake/max_risk_pct arguments
                  False -> switch the live risk rules off
                  a RiskConfig instance -> use that
  max_markup_pct  "live" -> config.MAX_MARKUP_PCT (15.0 if config can't be loaded)
                  None   -> premium gate off
                  number -> that limit

Minor simplification: a trade's result is booked on the day it was entered.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.cost_model import LEVELS, CostModel
from indicators import atr
from pricing import SECONDS_PER_YEAR, bs_price, index_vol

try:
    from staking import floor_cents
except ImportError:  # pragma: no cover
    def floor_cents(x: float) -> float:
        return math.floor(x * 100 + 1e-9) / 100


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
    gate_rejected: int = 0                         # signals skipped by the premium gate
    risk_rejected: dict = field(default_factory=dict)  # reason -> count, signals skipped by risk rules
    risk_applied: bool = False

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])


# --------------------------------------------------------------------- helpers
def _resolve_risk(risk_cfg):
    if risk_cfg is False:
        return None
    if risk_cfg is not None:
        return risk_cfg
    try:
        from risk import RiskConfig
        return RiskConfig.from_settings()
    except Exception:  # noqa: BLE001  config or risk module not importable here
        return None


def _resolve_markup_limit(value):
    if isinstance(value, str) and value == "live":
        try:
            import config
            return float(config.MAX_MARKUP_PCT)
        except Exception:  # noqa: BLE001
            return 15.0
    return value


def _size_stake(plan, balance: float, min_stake: float, max_stake: float, max_risk_pct: float) -> float:
    """Same sizing as RiskManager._size_stake."""
    wanted = plan.next_stake(balance)
    cap_pct = balance * max_risk_pct / 100
    stake = floor_cents(min(wanted, max_stake, cap_pct))
    if stake < min_stake <= min(max_stake, cap_pct):
        stake = min_stake
    return stake


class _RiskSim:
    """Replays RiskManager.approve_trade's limits against historical time."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.day = None
        self.day_start = None
        self.pnl = 0.0
        self.trades = 0
        self.consec = 0
        self.halted = False
        self.last_entry = None

    def _roll(self, ts: float) -> None:
        day = int(ts // 86400)
        if day != self.day:
            self.day = day
            self.day_start = None
            self.pnl = 0.0
            self.trades = 0
            self.consec = 0
            self.halted = False

    def approve(self, ts: float, balance: float):
        """None if allowed, otherwise a short reason."""
        self._roll(ts)
        cfg = self.cfg
        if self.day_start is None:
            self.day_start = balance - self.pnl
        if self.halted:
            return "halted for the rest of the day"
        if self.pnl <= -(self.day_start * cfg.daily_loss_limit_pct / 100):
            self.halted = True
            return "daily loss limit"
        if self.consec >= cfg.max_consecutive_losses:
            self.halted = True
            return "consecutive losses limit"
        if self.trades >= cfg.max_trades_per_day:
            return "daily trade limit"
        if (cfg.min_seconds_between_trades > 0 and self.last_entry is not None
                and ts - self.last_entry < cfg.min_seconds_between_trades):
            return "cooling down"
        return None

    def opened(self, ts: float) -> None:
        self.trades += 1
        self.last_entry = ts

    def closed(self, profit: float) -> None:
        self.pnl += profit
        self.consec = self.consec + 1 if profit < 0 else 0


# ------------------------------------------------------------------- the engine
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
    max_markup_pct="live",
    max_candidates: int = 3,
    risk_cfg=None,
) -> BacktestResult:
    epoch = candles["epoch"].to_numpy()
    close = candles["close"].to_numpy(dtype=float)
    n = len(close)
    gran = int(np.median(np.diff(epoch)))
    sigma = index_vol(symbol)
    hold = strategy.hold_candles
    seconds = hold * gran
    T = seconds / SECONDS_PER_YEAR

    cfg = _resolve_risk(risk_cfg)
    if cfg is not None:
        min_stake, max_stake, max_risk_pct = cfg.min_stake, cfg.max_stake, cfg.max_risk_pct_per_trade
    sim = _RiskSim(cfg) if cfg is not None else None
    markup_limit = _resolve_markup_limit(max_markup_pct)

    signals = strategy.signal_series(candles).to_numpy()
    atr_now = atr(candles, strategy.atr_n).to_numpy()

    balance = start_balance
    result = BacktestResult(balances=[balance], hold_candles=hold, strategy=strategy.name,
                            risk_applied=sim is not None)
    open_until = 0

    for i in np.flatnonzero(signals):
        if i < open_until:
            continue
        j = i + hold
        if j >= n:
            break
        if epoch[j] - epoch[i] != seconds or math.isnan(atr_now[i]):
            continue  # a gap in the data, or ATR not ready

        # 1. risk approval (same order as live)
        if sim is not None:
            reason = sim.approve(float(epoch[i]), balance)
            if reason:
                result.risk_rejected[reason] = result.risk_rejected.get(reason, 0) + 1
                continue

        direction = int(signals[i])
        kind, opposite = ("call", "put") if direction > 0 else ("put", "call")
        spot = close[i]
        unit = spot * sigma * math.sqrt(T)
        wanted = strategy.strike_atr * atr_now[i]

        # 2. candidate strike levels nearest the wanted strike; take the cheapest (lowest markup)
        ranked = sorted(LEVELS, key=lambda lv: abs(lv * unit - wanted))[:max(1, max_candidates)]
        markup_pct, level = min((cost_model.markup_pct(seconds, lv), lv) for lv in ranked)

        # 3. premium gate
        if markup_limit is not None and markup_pct > markup_limit:
            result.gate_rejected += 1
            continue

        strike = spot + direction * level * unit
        strike_opp = spot - direction * level * unit
        fair = bs_price(kind, spot, strike, T, sigma)
        fair_opp = bs_price(opposite, spot, strike_opp, T, sigma)
        if fair < 1e-9 or fair_opp < 1e-9:
            continue

        markup = markup_pct / 100
        stake = _size_stake(plan, balance, min_stake, max_stake, max_risk_pct)
        if stake < min_stake:
            reason = "stake below minimum"
            result.risk_rejected[reason] = result.risk_rejected.get(reason, 0) + 1
            continue

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
        if sim is not None:
            sim.opened(float(epoch[i]))
            sim.closed(profit)
        open_until = j
    return result