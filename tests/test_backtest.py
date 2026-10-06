import csv
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.cost_model import LEVELS, MEASURED, CostModel, load_markup_csv
from backtest.engine import Trade, run_backtest
from backtest.metrics import direction_test, summarize
from pricing import SECONDS_PER_YEAR, bs_call
from staking import FixedStake

ZERO = CostModel({60: [0.0] * 5})
ONE_MIN_MEASURED = CostModel({60: MEASURED[60]})


def candles_from_closes(closes, gap_at=None):
    closes = np.array(closes, dtype=float)
    epoch = 1_700_000_000 + 60 * np.arange(len(closes))
    if gap_at is not None:
        epoch[gap_at:] += 60                      # one missing candle
    return pd.DataFrame({"epoch": epoch, "open": closes, "high": closes + 0.5,
                         "low": closes - 0.5, "close": closes})


class Stub:
    """A strategy with hand-picked signals."""
    name = "stub"
    atr_n = 14

    def __init__(self, signals, hold=1, strike_atr=0.0):
        self._signals, self.hold_candles, self.strike_atr = np.array(signals), hold, strike_atr

    def signal_series(self, candles):
        return pd.Series(self._signals, index=candles.index)


def random_walk(n=40_000, seed=1, sigma=1.0):
    rng = np.random.default_rng(seed)
    dt = 60 / SECONDS_PER_YEAR
    log_ret = rng.normal(-0.5 * sigma**2 * dt, sigma * math.sqrt(dt), n)   # zero-drift price
    return candles_from_closes(1000 * np.exp(np.cumsum(log_ret)))


# ------------------------------------------------------------------ cost model
def test_markup_matches_measurements_at_table_points():
    cm = CostModel()
    assert cm.markup_pct(60, 0.0) == 21.0
    assert cm.markup_pct(300, 1.29) == 24.9
    assert cm.markup_pct(14400, -1.29) == 1.1


def test_markup_falls_with_expiry_and_is_clamped_outside_the_table():
    cm = CostModel()
    atm = [cm.markup_pct(s, 0.0) for s in (60, 300, 900, 3600, 14400)]
    assert atm == sorted(atm, reverse=True)
    assert cm.markup_pct(10, 0.0) == cm.markup_pct(60, 0.0)
    assert cm.markup_pct(10 * 86400, 0.0) == cm.markup_pct(14400, 0.0)
    assert MEASURED[300][2] < cm.markup_pct(180, 0.0) < MEASURED[60][2]


def test_markup_grows_as_the_strike_moves_out_of_the_money():
    cm = CostModel()
    values = [cm.markup_pct(300, lv) for lv in LEVELS]
    assert values == sorted(values)


def test_load_markup_csv_reads_levels_and_averages_calls_and_puts():
    spot = 1237.33
    unit = spot * math.sqrt(60 / SECONDS_PER_YEAR)
    rows = []
    for typ, sign in (("VANILLALONGCALL", 1), ("VANILLALONGPUT", -1)):
        for level, markup in zip(LEVELS, [10.0, 14.0, 21.0, 31.0, 42.0]):
            offset = sign * level * unit                       # CALL: OTM is above spot
            rows.append({"type": typ, "duration": "1m", "spot": spot,
                         "strike": spot + offset, "markup_pct": markup})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        table = load_markup_csv(path)
    assert table == {60: [10.0, 14.0, 21.0, 31.0, 42.0]}


# ---------------------------------------------------------------------- engine
def test_trade_profit_follows_the_option_payoff_formula():
    closes = [1000.0] * 40
    closes[21] = 1002.0
    signals = [0] * 40
    signals[20] = 1
    res = run_backtest(candles_from_closes(closes), Stub(signals), "1HZ100V", ZERO, FixedStake(1.0))
    assert len(res.trades) == 1
    T = 60 / SECONDS_PER_YEAR
    fair = bs_call(1000.0, 1000.0, T, 1.0)
    expected_profit = (1 / fair) * 2.0 - 1.0
    assert abs(res.trades[0].profit - expected_profit) < 1e-9
    assert res.balances[-1] == 1000.0 + res.trades[0].profit


def test_losing_trade_loses_exactly_the_stake():
    closes = [1000.0] * 40
    closes[21] = 990.0
    signals = [0] * 40
    signals[20] = 1
    res = run_backtest(candles_from_closes(closes), Stub(signals), "1HZ100V", ZERO, FixedStake(2.0))
    assert abs(res.trades[0].profit + 2.0) < 1e-9


def test_only_one_trade_at_a_time():
    signals = [0] * 40
    for i in (20, 21, 22, 23):
        signals[i] = 1
    res = run_backtest(candles_from_closes([1000.0] * 40), Stub(signals, hold=3), "1HZ100V",
                       ZERO, FixedStake(1.0))
    assert [t.index for t in res.trades] == [20, 23]


def test_gap_in_the_data_skips_the_trade():
    signals = [0] * 40
    signals[20] = 1
    res = run_backtest(candles_from_closes([1000.0] * 40, gap_at=21), Stub(signals), "1HZ100V",
                       ZERO, FixedStake(1.0))
    assert res.trades == []


def test_random_trading_at_fair_prices_has_no_edge():
    candles = random_walk()
    rng = np.random.default_rng(7)
    signals = np.where(rng.random(len(candles)) < 0.2, rng.choice([-1, 1], len(candles)), 0)
    res = run_backtest(candles, Stub(signals), "1HZ100V", ZERO, FixedStake(1.0), start_balance=1e6)
    m = summarize(res.trades, res.balances)
    assert m["trades"] > 5000
    assert abs(m["roi"]) < 0.06           # pricing engine and random walk agree: fair game


def test_the_markup_is_what_costs_you():
    candles = random_walk()
    rng = np.random.default_rng(7)
    signals = np.where(rng.random(len(candles)) < 0.2, rng.choice([-1, 1], len(candles)), 0)
    res = run_backtest(candles, Stub(signals), "1HZ100V", ONE_MIN_MEASURED, FixedStake(1.0), start_balance=1e6)
    m = summarize(res.trades, res.balances)
    assert m["roi"] < -0.10               # 21% markup on a 1.21x price is about -17%
    assert abs(m["gross_roi"]) < 0.06     # before the markup it is still a fair game


# --------------------------------------------------------------------- metrics
def make_trade(profit, stake=1.0, roi_opposite=0.0):
    return Trade(index=0, epoch=0, direction=1, level=0.0, spot=1.0, strike=1.0, expiry_price=1.0,
                 stake=stake, markup_pct=10.0, fair_cost=stake / 1.1, payoff=profit + stake,
                 profit=profit, roi=profit / stake, roi_opposite=roi_opposite)


def test_summary_numbers():
    trades = [make_trade(p) for p in (10, -5, -5, -5, 10, -5)]
    balances = [1000, 1010, 1005, 1000, 995, 1005, 1000]
    m = summarize(trades, balances)
    assert m["trades"] == 6 and m["total_profit"] == 0
    assert m["win_rate"] == 2 / 6
    assert m["longest_losing_streak"] == 3
    assert m["max_drawdown"] == 15.0
    assert abs(m["max_drawdown_pct"] - 15 / 1010 * 100) < 1e-9
    assert m["profit_factor"] == 20 / 20
    assert summarize([], [1000]) == {"trades": 0}


def test_direction_test_detects_a_strategy_that_knows_the_future():
    candles = random_walk(n=20_000, seed=3)
    close = candles["close"].to_numpy()
    rng = np.random.default_rng(5)
    pick = rng.random(len(close)) < 0.2
    future_up = np.r_[close[1:] > close[:-1], False]
    signals = np.where(pick, np.where(future_up, 1, -1), 0)            # cheating on purpose
    res = run_backtest(candles, Stub(signals), "1HZ100V", ZERO, FixedStake(1.0), start_balance=1e6)
    t = direction_test(res.trades, shuffles=500)
    assert t["p_value"] < 0.01 and t["edge_over_random"] > 0.5


def test_direction_test_does_not_flatter_random_signals():
    candles = random_walk(n=20_000, seed=3)
    rng = np.random.default_rng(11)
    signals = np.where(rng.random(len(candles)) < 0.2, rng.choice([-1, 1], len(candles)), 0)
    res = run_backtest(candles, Stub(signals), "1HZ100V", ZERO, FixedStake(1.0), start_balance=1e6)
    assert direction_test(res.trades, shuffles=500)["p_value"] > 0.01