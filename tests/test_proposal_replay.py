import numpy as np
import pandas as pd

from backtest.proposal_replay import load_proposals, replay, replay_report
from backtest.walkforward import blocksize_for

GRAN = 60
START = 1_700_000_000


def candles(closes, gap_at=None):
    closes = np.array(closes, dtype=float)
    epoch = START + GRAN * np.arange(len(closes))
    if gap_at is not None:
        epoch[gap_at:] += GRAN
    return pd.DataFrame({"epoch": epoch, "open": closes, "high": closes + 0.5,
                         "low": closes - 0.5, "close": closes})


PROPOSAL_KEYS = ("epoch", "contracttype", "duration", "durationunit", "barrier",
                 "spot", "askprice", "payoutperpoint")


def proposals(rows):
    """rows: (epoch, contracttype, duration, durationunit, barrier, spot, askprice, payout)."""
    return pd.DataFrame(
        [{**dict(zip(PROPOSAL_KEYS, r)), "symbol": "1HZ100V", "payout": 0.0,
          "proposalid": f"p{i}"} for i, r in enumerate(rows)]
    )


def write(tmp_path, frame, name="proposals.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_relative_plus_barrier_keeps_its_sign(tmp_path):
    """Regression: pandas infers `barrier` as float and drops the `+`, which would
    silently price a `+0.00` barrier at a strike of 0 instead of the spot."""
    path = write(tmp_path, proposals(
        [(START + 5 * GRAN, "VANILLALONGPUT", 5, "m", "+0.00", 100.0, 25.0, 40.0)]))
    loaded = load_proposals(path)
    assert loaded["barrier"].iloc[0] == "+0.00"

    settled = replay(loaded, candles([100.0] * 20))
    assert settled["strike"].iloc[0] == 100.0


def test_relative_minus_barrier_is_offset_from_spot(tmp_path):
    path = write(tmp_path, proposals(
        [(START, "VANILLALONGPUT", 5, "m", "-5.00", 100.0, 25.0, 40.0)]))
    settled = replay(load_proposals(path), candles([100.0] * 20))
    assert settled["strike"].iloc[0] == 95.0


def test_profit_follows_the_observed_quote(tmp_path):
    closes = [100.0] * 5 + [80.0] + [80.0] * 10
    path = write(tmp_path, proposals(
        [(START, "VANILLALONGPUT", 5, "m", "-5.00", 100.0, 25.0, 40.0)]))
    settled = replay(load_proposals(path), candles(closes))

    row = settled.iloc[0]
    strike, final = 95.0, 80.0
    payoff = 40.0 * max(strike - final, 0.0)
    assert row["payoff"] == payoff
    assert row["profit"] == payoff - 25.0
    assert row["roi"] == (payoff - 25.0) / 25.0


def test_expired_worthless_contract_costs_the_premium(tmp_path):
    closes = [100.0] * 5 + [100.0] + [100.0] * 10
    path = write(tmp_path, proposals(
        [(START, "VANILLALONGPUT", 5, "m", "-5.00", 100.0, 25.0, 40.0)]))
    settled = replay(load_proposals(path), candles(closes))
    assert settled["profit"].iloc[0] == -25.0
    assert settled["roi"].iloc[0] == -1.0


def test_quote_past_the_end_of_the_candles_is_not_settled(tmp_path):
    path = write(tmp_path, proposals(
        [(START, "VANILLALONGPUT", 5, "m", "-5.00", 100.0, 25.0, 40.0)]))
    assert replay(load_proposals(path), candles([100.0] * 5)).empty


def test_a_gap_in_the_candles_is_not_guessed(tmp_path):
    closes = [100.0] * 5 + [80.0] * 10
    path = write(tmp_path, proposals(
        [(START, "VANILLALONGPUT", 5, "m", "-5.00", 100.0, 25.0, 40.0)]))
    # one candle missing three minutes after the entry shifts the settlement point
    assert replay(load_proposals(path), candles(closes, gap_at=3)).empty


def test_unknown_duration_unit_and_missing_file(tmp_path):
    bad = write(tmp_path, proposals(
        [(START, "VANILLALONGPUT", 5, "w", "-5.00", 100.0, 25.0, 40.0)]))
    assert load_proposals(bad).empty
    assert load_proposals(tmp_path / "nope.csv").empty
    assert replay_report(tmp_path / "nope.csv", candles([100.0] * 20), "1HZ100V") is None


def test_report_flags_a_positive_confidence_interval(tmp_path):
    # five quotes whose puts all finish in the money
    rows = [(START + i * GRAN, "VANILLALONGPUT", 5, "m", "-5.00", 100.0, 25.0, 40.0)
            for i in range(5)]
    path = write(tmp_path, proposals(rows))
    closes = [100.0] * 5 + [80.0] * 40
    info = replay_report(path, candles(closes), "1HZ100V")
    assert info["quotes"] == 5 and info["settled"] == 5
    assert info["profitable"] is True
    assert info["cilow"] > 0.0


def test_report_counts_quotes_it_could_not_settle(tmp_path):
    rows = [(START + i * GRAN, "VANILLALONGPUT", 5, "m", "-5.00", 100.0, 25.0, 40.0)
            for i in range(3)]
    path = write(tmp_path, proposals(rows))
    info = replay_report(path, candles([100.0] * 4), "1HZ100V")
    assert info["quotes"] == 3 and info["settled"] == 0


def test_symbol_filter(tmp_path):
    rows = [(START, "VANILLALONGPUT", 5, "m", "-5.00", 100.0, 25.0, 40.0)]
    frame = proposals(rows)
    path = write(tmp_path, frame)
    assert load_proposals(path, symbol="1HZ101V").empty


def test_blocksize_never_collapses_to_a_single_block():
    assert blocksize_for(5, 0) == 1
    assert blocksize_for(5, 1) == 1
    assert blocksize_for(5, 8) == 2          # capped by n // 4, so at least 4 blocks
    assert blocksize_for(5, 100) == 5        # the holding period, once there is enough data
