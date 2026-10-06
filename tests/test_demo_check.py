import asyncio
import tempfile
from pathlib import Path

from demo_trade_check import check
from pricing import bs_call, years


class FakeClient:
    def __init__(self):
        self.bought = []

    async def get_balance(self):
        return {"balance": 9000.0, "currency": "USD"}

    async def subscribe_ticks(self, symbol, cb):
        return {"tick": {"quote": 1000.0}, "subscription": {"id": "t1"}}

    async def forget(self, sub_id):
        pass

    async def get_allowed_barriers(self, *args):
        return ["+1.10", "+0.00", "-1.10"]

    async def get_proposal(self, ctype, symbol, barrier, duration, unit, stake, currency):
        fair = bs_call(1000.0, 1000.0, years(duration, unit), 1.0)
        return {"id": "p1", "ask_price": stake, "display_number_of_contracts": str(stake / (fair * 1.1)),
                "contract_details": {"barrier": "1000.00"}, "spot": 1000.0}

    async def buy(self, proposal_id, max_price):
        self.bought.append(proposal_id)
        return {"contract_id": 42, "buy_price": max_price}

    async def subscribe(self, payload, callback):
        asyncio.get_running_loop().call_later(
            0.01, callback, {"proposal_open_contract": {"is_sold": 1, "profit": "-0.5", "status": "lost"}}
        )
        return {"proposal_open_contract": {"is_sold": 0}, "subscription": {"id": "s1"}}


def test_cancelled_check_buys_nothing():
    async def go():
        client = FakeClient()
        result = await check(client, "1HZ100V", 0.5, 60, confirm=lambda a, b: False)
        assert result == {"cancelled": True} and client.bought == []

    asyncio.run(go())


def test_confirmed_check_buys_once_and_records_the_settlement():
    async def go():
        client = FakeClient()
        with tempfile.TemporaryDirectory() as tmp:
            record = await check(client, "1HZ100V", 0.5, 60, confirm=lambda a, b: True, out_dir=Path(tmp))
            assert len(list(Path(tmp).glob("demo_trade_check_*.json"))) == 1
        assert client.bought == ["p1"]
        assert record["buy"]["contract_id"] == 42
        assert record["final_open_contract"]["profit"] == "-0.5"

    asyncio.run(go())
