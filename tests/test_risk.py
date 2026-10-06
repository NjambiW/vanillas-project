import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pricing import bs_call, years
from risk import PremiumGate, RiskConfig, RiskManager
from staking import FixedFraction, FixedStake


class Clock:
    def __init__(self):
        self.t = datetime(2026, 10, 6, 9, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


def manager(cfg=None, staking=None, clock=None):
    return RiskManager(cfg or RiskConfig(), staking or FixedFraction(0.01), now=clock or Clock())


def test_stake_is_fraction_of_balance():
    d = manager().approve_trade(1000)
    assert d.allowed and d.stake == 10.0


def test_stake_capped_by_percent_of_balance_and_max_stake():
    plan = FixedStake(50)
    d = manager(RiskConfig(max_risk_pct_per_trade=2.0, max_stake=100), plan).approve_trade(1000)
    assert d.stake == 20.0                     # 2% of 1000
    d = manager(RiskConfig(max_risk_pct_per_trade=50, max_stake=15), plan).approve_trade(1000)
    assert d.stake == 15.0                     # max_stake


def test_stake_raised_to_minimum_when_allowed():
    d = manager(staking=FixedFraction(0.0001)).approve_trade(1000)
    assert d.allowed and d.stake == 1.0


def test_denied_when_balance_too_small():
    d = manager().approve_trade(10)            # 2% of 10 is 0.20, below the 1.00 minimum
    assert not d.allowed and "minimum" in d.reason


def test_daily_loss_limit_halts_until_next_day():
    clock = Clock()
    m = manager(clock=clock)
    assert m.approve_trade(1000).allowed       # day starts at 1000, limit is 30
    for _ in range(3):
        m.on_trade_opened(10)
        m.on_trade_closed(-10)
    d = m.approve_trade(970)
    assert not d.allowed and "daily loss limit" in d.reason
    assert not m.approve_trade(970).allowed    # stays halted
    clock.advance(days=1)
    assert m.approve_trade(970).allowed        # new UTC day, fresh start


def test_consecutive_losses_halt_and_a_win_resets_the_streak():
    cfg = RiskConfig(max_consecutive_losses=3, daily_loss_limit_pct=100)
    m = manager(cfg)
    for _ in range(2):
        m.on_trade_opened(1)
        m.on_trade_closed(-1)
    m.on_trade_opened(1)
    m.on_trade_closed(+5)                      # streak broken
    for _ in range(2):
        m.on_trade_opened(1)
        m.on_trade_closed(-1)
    assert m.approve_trade(1000).allowed
    m.on_trade_opened(1)
    m.on_trade_closed(-1)
    d = m.approve_trade(1000)
    assert not d.allowed and "in a row" in d.reason


def test_only_one_open_position_by_default():
    m = manager()
    assert m.approve_trade(1000).allowed
    m.on_trade_opened(10)
    assert not m.approve_trade(1000).allowed
    m.on_trade_closed(5)
    assert m.approve_trade(1000).allowed


def test_max_trades_per_day():
    m = manager(RiskConfig(max_trades_per_day=2))
    for _ in range(2):
        m.on_trade_opened(1)
        m.on_trade_closed(1)
    d = m.approve_trade(1000)
    assert not d.allowed and "trade limit" in d.reason


def test_cooldown_between_trades():
    clock = Clock()
    m = manager(RiskConfig(min_seconds_between_trades=60), clock=clock)
    m.on_trade_opened(1)
    m.on_trade_closed(1)
    assert not m.approve_trade(1000).allowed
    clock.advance(seconds=61)
    assert m.approve_trade(1000).allowed


def test_kill_file_stops_trading():
    with tempfile.TemporaryDirectory() as tmp:
        kill = Path(tmp) / "STOP"
        m = manager(RiskConfig(kill_file=kill))
        assert m.approve_trade(1000).allowed
        kill.write_text("")
        d = m.approve_trade(1000)
        assert not d.allowed and "kill file" in d.reason
        kill.unlink()
        assert m.approve_trade(1000).allowed


def test_manual_halt():
    m = manager()
    m.halt("connection errors")
    d = m.approve_trade(1000)
    assert not d.allowed and "connection errors" in d.reason


def _proposal(markup):
    spot, k = 1000.0, 0.8
    fair = k * bs_call(spot, spot, years(5, "m"), 1.0)
    return {"ask_price": fair * (1 + markup / 100), "payout_per_point": k, "spot": spot}


def test_premium_gate_accepts_and_rejects_by_markup():
    gate = PremiumGate(max_markup_pct=10)
    ok = gate.check("VANILLALONGCALL", _proposal(5), 1000.0, "+0.00", 5, "m", "1HZ100V")
    bad = gate.check("VANILLALONGCALL", _proposal(25), 1000.0, "+0.00", 5, "m", "1HZ100V")
    assert ok.allowed and abs(ok.markup_pct - 5) < 0.01
    assert not bad.allowed and "above" in bad.reason


def test_premium_gate_fails_closed_on_unreadable_proposal():
    gate = PremiumGate(max_markup_pct=100)
    d = gate.check("VANILLALONGCALL", {"ask_price": 1.0}, 1000.0, "+0.00", 5, "m", "1HZ100V")
    assert not d.allowed and "cannot verify" in d.reason


def test_release_position_frees_the_slot_without_recording_a_result():
    m = manager()
    m.on_trade_opened(10)
    assert not m.approve_trade(1000).allowed
    m.release_position()
    assert m.approve_trade(1000).allowed
    assert m.pnl_today == 0