import asyncio
import json

import pytest

from deriv_client import DerivAPIError, DerivClient


class FakeWS:
    """Stands in for a websocket: records what we send, replays what we queue."""

    def __init__(self):
        self.sent = []
        self.inbox = asyncio.Queue()

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.inbox.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def close(self):
        await self.inbox.put(None)

    def reply(self, **msg):
        self.inbox.put_nowait(json.dumps(msg))


async def make_client():
    client = DerivClient(url_provider=lambda: "wss://unused")
    client._ws = FakeWS()
    client._connected.set()
    client._reader = asyncio.create_task(client._read_loop())
    await asyncio.sleep(0)
    return client


def run(coro):
    return asyncio.run(coro)


def test_request_gets_matching_response():
    async def go():
        c = await make_client()
        task = asyncio.create_task(c.request({"balance": 1}))
        await asyncio.sleep(0.01)
        rid = c._ws.sent[0]["req_id"]
        c._ws.reply(req_id=rid, msg_type="balance", balance={"balance": 100, "currency": "USD"})
        assert (await task)["balance"]["currency"] == "USD"

    run(go())


def test_api_error_raises():
    async def go():
        c = await make_client()
        task = asyncio.create_task(c.request({"proposal": 1}))
        await asyncio.sleep(0.01)
        rid = c._ws.sent[0]["req_id"]
        c._ws.reply(req_id=rid, error={"code": "ContractBuyValidationError", "message": "bad barrier"})
        with pytest.raises(DerivAPIError, match="bad barrier"):
            await task

    run(go())


def test_unknown_req_id_is_ignored():
    async def go():
        c = await make_client()
        c._ws.reply(req_id=999, msg_type="tick", tick={"quote": 1})
        await asyncio.sleep(0.01)  # must not crash the reader
        assert not c._reader.done()

    run(go())


def test_subscription_delivers_later_messages():
    async def go():
        c = await make_client()
        seen = []
        task = asyncio.create_task(c.subscribe_ticks("1HZ100V", seen.append))
        await asyncio.sleep(0.01)
        sent = c._ws.sent[0]
        assert sent["ticks"] == "1HZ100V" and sent["subscribe"] == 1
        rid = sent["req_id"]
        c._ws.reply(req_id=rid, msg_type="tick", tick={"quote": 1.0}, subscription={"id": "s1"})
        first = await task
        assert first["subscription"]["id"] == "s1"
        c._ws.reply(req_id=rid, msg_type="tick", tick={"quote": 2.0}, subscription={"id": "s1"})
        await asyncio.sleep(0.01)
        assert [m["tick"]["quote"] for m in seen] == [1.0, 2.0]

    run(go())


def test_forget_stops_resubscribing():
    async def go():
        c = await make_client()
        task = asyncio.create_task(c.subscribe_ticks("1HZ100V", lambda m: None))
        await asyncio.sleep(0.01)
        rid = c._ws.sent[0]["req_id"]
        c._ws.reply(req_id=rid, msg_type="tick", tick={"quote": 1.0}, subscription={"id": "s1"})
        await task
        assert rid in c._subs
        ftask = asyncio.create_task(c.forget("s1"))
        await asyncio.sleep(0.01)
        c._ws.reply(req_id=c._ws.sent[1]["req_id"], msg_type="forget", forget=1)
        await ftask
        assert rid not in c._subs

    run(go())


def test_proposal_payload_for_vanilla_call():
    async def go():
        c = await make_client()
        task = asyncio.create_task(
            c.get_proposal("VANILLALONGCALL", "1HZ100V", "+1.50", 5, "m", 2.0, "USD")
        )
        await asyncio.sleep(0.01)
        sent = c._ws.sent[0]
        assert sent["proposal"] == 1
        assert sent["contract_type"] == "VANILLALONGCALL"
        assert sent["underlying_symbol"] == "1HZ100V"
        assert sent["barrier"] == "+1.50"
        assert sent["basis"] == "stake" and sent["amount"] == 2.0
        assert sent["duration"] == 5 and sent["duration_unit"] == "m"
        c._ws.reply(req_id=sent["req_id"], msg_type="proposal", proposal={"id": "p1", "ask_price": 2.0})
        assert (await task)["id"] == "p1"

    run(go())


def test_pending_requests_fail_when_connection_drops():
    async def go():
        c = await make_client()
        task = asyncio.create_task(c.request({"balance": 1}))
        await asyncio.sleep(0.01)
        await c._ws.close()  # simulates the socket closing
        with pytest.raises(ConnectionError):
            await task

    run(go())


def test_allowed_barriers_are_read_from_the_error_message():
    async def go():
        c = await make_client()
        task = asyncio.create_task(
            c.get_allowed_barriers("VANILLALONGCALL", "1HZ100V", 5, "m", 1.0, "USD")
        )
        await asyncio.sleep(0.01)
        rid = c._ws.sent[0]["req_id"]
        c._ws.reply(
            req_id=rid,
            error={
                "code": "ContractBuyValidationError",
                "message": "Barriers available are +4.40, +2.30, +0.00, -2.30, -4.40.",
            },
        )
        assert await task == ["+4.40", "+2.30", "+0.00", "-2.30", "-4.40"]

    run(go())


def test_allowed_barriers_can_be_absolute_strikes():
    async def go():
        c = await make_client()
        task = asyncio.create_task(
            c.get_allowed_barriers("VANILLALONGCALL", "1HZ100V", 1, "d", 1.0, "USD")
        )
        await asyncio.sleep(0.01)
        rid = c._ws.sent[0]["req_id"]
        c._ws.reply(
            req_id=rid,
            error={
                "code": "ContractBuyValidationError",
                "message": "Barriers available are 1140.00, 1160.00, 1350.00.",
            },
        )
        assert await task == ["1140.00", "1160.00", "1350.00"]

    run(go())