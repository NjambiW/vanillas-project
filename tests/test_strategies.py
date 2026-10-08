import numpy as np
import pandas as pd
import pytest

from strategies import STRATEGIES, Breakout, MeanReversion, Signal, TrendPullback, get_strategy


def make_candles(closes):
    c = np.array(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame(
        {"open": o, "high": np.maximum(o, c) + 0.5, "low": np.minimum(o, c) - 0.5, "close": c}
    )


def squeeze_then_jump(jump):
    """Quiet, narrowing market for 170 candles, then one big candle."""
    rng = np.random.default_rng(0)
    n = 170
    closes = list(100 + rng.normal(0, 1, n) * np.linspace(1.0, 0.02, n))
    closes.append(closes[-1] + jump)
    return make_candles(closes)


def ranging_then_move(direction):
    rng = np.random.default_rng(1)
    flat = list(100 + rng.normal(0, 0.5, 150))
    steps = [flat[-1] + direction * 1.5 * i for i in (1, 2, 3)]
    return make_candles(flat + steps)


def test_trend_pullback_calls_when_price_reclaims_ema_in_uptrend():
    ramp = [100 + 0.5 * i for i in range(120)]
    top = ramp[-1]
    closes = ramp + [top - 3, top - 6, top - 9] + [top - 3, top + 1, top + 3]
    signals = TrendPullback().signal_series(make_candles(closes))
    assert signals.iloc[123] == 1
    assert (signals == -1).sum() == 0


def test_trend_pullback_puts_in_downtrend():
    ramp = [200 - 0.5 * i for i in range(120)]
    bottom = ramp[-1]
    closes = ramp + [bottom + 3, bottom + 6, bottom + 9] + [bottom + 3, bottom - 1, bottom - 3]
    signals = TrendPullback().signal_series(make_candles(closes))
    assert signals.iloc[123] == -1
    assert (signals == 1).sum() == 0


def test_breakout_up_and_down():
    up = Breakout().signal_series(squeeze_then_jump(+3))
    down = Breakout().signal_series(squeeze_then_jump(-3))
    assert up.iloc[-1] == 1 and (up.iloc[:-1] == 0).all()
    assert down.iloc[-1] == -1 and (down.iloc[:-1] == 0).all()


def test_mean_reversion_fades_stretches_in_a_range():
    assert MeanReversion().signal_series(ranging_then_move(-1)).iloc[-1] == 1
    assert MeanReversion().signal_series(ranging_then_move(+1)).iloc[-1] == -1


def test_no_signals_before_warmup():
    candles = squeeze_then_jump(+3)
    for name in STRATEGIES:
        strat = get_strategy(name)
        assert (strat.signal_series(candles).iloc[: strat.warmup] == 0).all()


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategy_never_looks_at_the_future(name):
    """Signals computed on a shortened history must equal the same rows of the full run."""
    rng = np.random.default_rng(5)
    candles = make_candles(100 + rng.normal(0, 1, 500).cumsum())
    strat = get_strategy(name)
    full = strat.signal_series(candles)
    for cut in (150, 250, 399, 450):
        part = strat.signal_series(candles.iloc[:cut])
        assert (full.iloc[:cut].to_numpy() == part.to_numpy()).all()


def test_evaluate_returns_a_signal():
    sig = Breakout().evaluate(squeeze_then_jump(+3))
    assert isinstance(sig, Signal)
    assert sig.direction == "CALL" and sig.contract_type == "VANILLALONGCALL"
    assert sig.hold_candles == 3 and sig.atr > 0
    assert sig.strike_distance == pytest.approx(0.7 * sig.atr)
    assert "squeeze" in sig.reason


def test_evaluate_put_contract_type():
    sig = Breakout().evaluate(squeeze_then_jump(-3))
    assert sig.direction == "PUT" and sig.contract_type == "VANILLALONGPUT"


def test_evaluate_is_none_without_a_setup_or_enough_data():
    candles = squeeze_then_jump(+3)
    assert Breakout().evaluate(candles.iloc[:-1]) is None   # jump candle not there yet
    assert Breakout().evaluate(candles.iloc[:30]) is None   # not enough history


def test_registry():
    assert isinstance(get_strategy("breakout"), Breakout)
    assert get_strategy("breakout", hold_candles=9).hold_candles == 9
    with pytest.raises(ValueError, match="unknown strategy"):
        get_strategy("nope")


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_live_window_gives_the_same_signal_as_the_full_history(name):
    """The live bot only sees a rolling window of recent candles. Its signal for the newest candle
    must equal what the backtest (which sees the whole history) said for that same candle."""
    rng = np.random.default_rng(5)
    candles = make_candles(100 + rng.normal(0, 1, 2500).cumsum())
    strat = get_strategy(name)
    full = strat.signal_series(candles).to_numpy()
    window = 500
    for end in range(window, len(candles) + 1, 7):
        live = strat.signal_series(candles.iloc[end - window:end]).to_numpy()[-1]
        assert live == full[end - 1], f"{name}: window and history disagree at candle {end - 1}"