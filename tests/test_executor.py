import asyncio

import pandas as pd

from deriv_client import DerivAPIError
from executor import Executor, barrier_offset, pick_barrier, seconds_to_duration
from journal import Journal
from pricing import bs_price, years
from risk import PremiumGate, RiskConfig, RiskManager
from staking import FixedFraction
from strategies.base import Signal


class FakeClient:
    """Stands in for DerivClient and records what the executor asked for."""

    def __init__(self, markup_pct=5.0):
        self.markup_pct = markup_pct
        self.proposal_args = []
        self.bought = []
        self.forgot = []
        self.sub_cb = None
        self.buy_error = None
        self.balance_error = None

    async def get_balance(self):
        if self.balance_error:
            raise self.balance_error
        return {"balance": 1000.0, "currency": "USD"}

    async def get_allowed_barriers(self, contract_type, symbol, duration, unit, stake, currency):
        return ["+2.20", "+1.10", "+0.00", "-1.10", "-2.20"]

    async def get_proposal(self, contract_type, symbol, barrier, duration, unit, stake, currency):
        self.proposal_args.append((contract_type, barrier, duration, unit, stake))
        spot, strike = 1000.0, 1000.0 + float(barrier)
        kind = "call" if contract_type.endswith("CALL") else "put"
        fair_points = bs_price(kind, spot, strike, years(duration, unit), 1.0)
        k = stake / (fair_points * (1 + self.markup_pct / 100))
        return {
            "id": "p1",
            "ask_price": stake,
            "display_number_of_contracts": str(k),
            "contract_details": {"barrier": f"{strike:.2f}"},
            "spot": spot,
            "min_stake": 0.4,
            "max_stake": 567,
        }

    async def buy(self, proposal_id, max_price):
        if self.buy_error:
            raise self.buy_error
        self.bought.append((proposal_id, max_price))
        return {"contract_id": 777, "buy_price": max_price}

    async def subscribe(self, payload, callback):
        self.sub_cb = callback
        return {"proposal_open_contract": {"is_sold": 0}, "subscription": {"id": "sub1"}}

    async def forget(self, sub_id):
        self.forgot.append(sub_id)

    async def request(self, payload, timeout=15):
        return {"proposal_open_contract": {"is_sold": 0}}


class StubStrategy:
    name = "stub"

    def __init__(self, signal):
        self.signal = signal

    def evaluate(self, candles):
        return self.signal


SIGNAL = Signal(direction="CALL", reason="test", atr=2.0, strike_atr=0.5, hold_candles=1)


def build(client=None, dry_run=False, max_markup=15.0, signal=SIGNAL):
    client = client or FakeClient()
    risk = RiskManager(RiskConfig(), FixedFraction(0.01))
    journal = Journal()
    ex = Executor(client, StubStrategy(signal), risk, PremiumGate(max_markup), journal,
                  "1HZ100V", 60, dry_run=dry_run, settle_buffer=0.05)
    return ex, client, risk, journal


def skip_texts(journal):
    return [r[0] for r in journal.db.execute("SELECT reason FROM skips")]


def test_pick_barrier_and_duration_helpers():
    allowed = ["+2.20", "+1.10", "+0.00", "-1.10", "-2.20"]
    assert pick_barrier(allowed, 1.0) == "+1.10"
    assert pick_barrier(allowed, -2.0) == "-2.20"
    assert pick_barrier(allowed, 0.1) == "+0.00"
    assert seconds_to_duration(300) == (5, "m")
    assert seconds_to_duration(3600) == (1, "h")
    assert seconds_to_duration(172800) == (2, "d")
    assert seconds_to_duration(45) == (45, "s")


def test_full_trade_from_signal_to_settlement():
    async def go():
        ex, client, risk, journal = build()
        trade_id = await ex.handle_signal(SIGNAL, 1000.0)
        assert trade_id is not None
        assert client.proposal_args == [("VANILLALONGCALL", "+1.10", 1, "m", 10.0)]
        assert client.bought == [("p1", 10.0)]
        await asyncio.sleep(0.01)                      # let the watcher subscribe
        client.sub_cb({"proposal_open_contract": {"is_sold": 1, "profit": "7.5"}})
        await ex.wait_idle()
        s = journal.summary()
        assert s["wins"] == 1 and s["total_profit"] == 7.5
        assert risk.pnl_today == 7.5
        assert client.forgot == ["sub1"]
        assert risk.approve_trade(1000).allowed        # slot is free again

    asyncio.run(go())


def test_put_signal_picks_a_barrier_below_spot():
    async def go():
        put = Signal(direction="PUT", reason="t", atr=2.0, strike_atr=0.5, hold_candles=1)
        ex, client, _, _ = build(signal=put)
        await ex.handle_signal(put, 1000.0)
        assert client.proposal_args[0][:2] == ("VANILLALONGPUT", "-1.10")

    asyncio.run(go())


def test_dry_run_never_buys():
    async def go():
        ex, client, risk, journal = build(dry_run=True)
        assert await ex.handle_signal(SIGNAL, 1000.0) is None
        assert client.bought == []
        assert any("DRY RUN" in t for t in skip_texts(journal))
        assert risk.approve_trade(1000).allowed

    asyncio.run(go())


def test_risk_denial_is_logged_and_nothing_is_priced():
    async def go():
        ex, client, risk, journal = build()
        risk.halt("testing halt")
        assert await ex.handle_signal(SIGNAL, 1000.0) is None
        assert client.proposal_args == [] and client.bought == []
        assert skip_texts(journal) == ["testing halt"]

    asyncio.run(go())


def test_expensive_price_is_rejected_by_the_gate():
    async def go():
        ex, client, _, journal = build(client=FakeClient(markup_pct=50.0), max_markup=10.0)
        assert await ex.handle_signal(SIGNAL, 1000.0) is None
        assert client.bought == []
        assert any("above" in t for t in skip_texts(journal))

    asyncio.run(go())


def test_rejected_buy_is_logged_and_does_not_block_the_next_trade():
    async def go():
        client = FakeClient()
        client.buy_error = DerivAPIError({"code": "PriceMoved", "message": "price moved"})
        ex, _, risk, journal = build(client=client)
        assert await ex.handle_signal(SIGNAL, 1000.0) is None
        assert any("buy rejected" in t for t in skip_texts(journal))
        assert risk.approve_trade(1000).allowed

    asyncio.run(go())


def test_unconfirmed_settlement_halts_the_bot_and_frees_the_slot():
    async def go():
        ex, client, risk, journal = build()
        risk.on_trade_opened(10)
        trade_id = journal.open_trade(strategy="stub", contract_id=5, stake=10.0)
        await ex._watch(5, trade_id, expiry_seconds=0)   # never reports settled
        assert journal.summary()["unknown"] == 1
        d = risk.approve_trade(1000)
        assert not d.allowed and "could not confirm" in d.reason

    asyncio.run(go())


def test_handle_candle_logs_the_signal_and_survives_errors():
    async def go():
        ex, client, risk, journal = build(dry_run=True)
        frame = pd.DataFrame({"close": [1000.0]})
        await ex.handle_candle(frame)
        assert journal.db.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == 1

        client.balance_error = RuntimeError("network down")
        for _ in range(3):
            await ex.handle_candle(frame)               # must not raise
        d = risk.approve_trade(1000)
        assert not d.allowed and "errors in a row" in d.reason

    asyncio.run(go())


def test_absolute_barriers_are_converted_to_offsets():
    assert barrier_offset("+2.20", 1237.0) == 2.2
    assert barrier_offset("-1.10", 1237.0) == -1.1
    assert abs(barrier_offset("1240.00", 1237.0) - 3.0) < 1e-9
    absolute = ["1200.00", "1220.00", "1240.00", "1260.00"]
    assert pick_barrier(absolute, 4.0, spot=1237.0) == "1240.00"
    assert pick_barrier(absolute, -15.0, spot=1237.0) == "1220.00"