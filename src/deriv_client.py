"""Async WebSocket client for the Deriv Options API.

Features: OTP authentication, request/response matching by req_id,
subscriptions (ticks etc.), automatic reconnect with backoff, and
re-subscribing after a reconnect.
"""
import asyncio
import itertools
import json
import logging
import re
from typing import Any, Callable, Optional

import websockets

from authenticator import get_websocket_url

log = logging.getLogger("deriv_client")


class DerivAPIError(Exception):
    """The API answered with an error object."""

    def __init__(self, error: dict):
        self.code = error.get("code")
        self.message = error.get("message", "unknown error")
        super().__init__(f"{self.code}: {self.message}")


class DerivClient:
    def __init__(self, url_provider: Callable[[], str] = get_websocket_url):
        self._url_provider = url_provider
        self._ws = None
        self._reader: Optional[asyncio.Task] = None
        self._supervisor: Optional[asyncio.Task] = None
        self._req_ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._subs: dict[int, tuple[dict, Callable[[dict], None]]] = {}
        self._sub_ids: dict[str, int] = {}  # subscription id -> req_id
        self._connected = asyncio.Event()
        self._closing = False

    # ------------------------------------------------------------------ life
    async def start(self, timeout: float = 30) -> None:
        """Connect and keep the connection alive in the background."""
        self._closing = False
        self._supervisor = asyncio.create_task(self._supervise())
        await asyncio.wait_for(self._connected.wait(), timeout)

    async def close(self) -> None:
        self._closing = True
        if self._supervisor:
            self._supervisor.cancel()
        if self._reader:
            self._reader.cancel()
        if self._ws:
            await self._ws.close()
        self._connected.clear()
        self._fail_pending(ConnectionError("client closed"))

    async def _supervise(self) -> None:
        delay = 1
        while not self._closing:
            try:
                await self._open()
                delay = 1
                await self._reader  # returns when the connection drops
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("connection problem: %s", exc)
            if self._closing:
                break
            log.info("reconnecting in %ss", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)

    async def _open(self) -> None:
        # The OTP URL is short-lived, so fetch a fresh one every time.
        url = await asyncio.to_thread(self._url_provider)
        self._ws = await websockets.connect(url, ping_interval=20, ping_timeout=20)
        self._reader = asyncio.create_task(self._read_loop())
        self._connected.set()
        for req_id, (payload, _cb) in list(self._subs.items()):
            await self._ws.send(json.dumps({**payload, "req_id": req_id}))
        log.info("connected")

    # --------------------------------------------------------------- reading
    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                self._dispatch(json.loads(raw))
        except websockets.ConnectionClosed:
            pass
        finally:
            self._connected.clear()
            self._fail_pending(ConnectionError("WebSocket closed"))

    def _dispatch(self, msg: dict) -> None:
        req_id = msg.get("req_id")
        fut = self._pending.pop(req_id, None)
        if fut is not None and not fut.done():
            if "error" in msg:
                fut.set_exception(DerivAPIError(msg["error"]))
            else:
                fut.set_result(msg)
        sub = self._subs.get(req_id)
        if sub is not None and "error" not in msg:
            try:
                sub[1](msg)
            except Exception:  # noqa: BLE001  a bad callback must not kill the reader
                log.exception("subscription callback failed")

    def _fail_pending(self, exc: Exception) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    # -------------------------------------------------------------- requests
    async def _send(self, req_id: int, payload: dict, timeout: float) -> dict:
        await asyncio.wait_for(self._connected.wait(), timeout)
        fut = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        await self._ws.send(json.dumps({**payload, "req_id": req_id}))
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise

    async def request(self, payload: dict, timeout: float = 15) -> dict:
        """Send one request and wait for its response."""
        return await self._send(next(self._req_ids), payload, timeout)

    async def subscribe(
        self, payload: dict, callback: Callable[[dict], None], timeout: float = 15
    ) -> dict:
        """Start a stream. Returns the first message; later ones go to callback."""
        req_id = next(self._req_ids)
        payload = {**payload, "subscribe": 1}
        self._subs[req_id] = (payload, callback)
        try:
            first = await self._send(req_id, payload, timeout)
        except Exception:
            self._subs.pop(req_id, None)
            raise
        sub_id = (first.get("subscription") or {}).get("id")
        if sub_id:
            self._sub_ids[sub_id] = req_id
        return first

    # ------------------------------------------------------------ convenience
    async def get_balance(self) -> dict:
        """Returns e.g. {'balance': 10000.0, 'currency': 'USD', ...}."""
        return (await self.request({"balance": 1}))["balance"]

    async def subscribe_ticks(self, symbol: str, callback: Callable[[dict], None]) -> dict:
        return await self.subscribe({"ticks": symbol}, callback)

    async def forget(self, subscription_id: str) -> None:
        """Stop a stream and stop re-subscribing to it after reconnects."""
        await self.request({"forget": subscription_id})
        req_id = self._sub_ids.pop(subscription_id, None)
        if req_id is not None:
            self._subs.pop(req_id, None)

    async def get_candles(
        self,
        symbol: str,
        granularity: int = 60,
        count: int = 500,
        end: str = "latest",
    ) -> list[dict]:
        """Historical candles, oldest first: [{epoch, open, high, low, close}, ...]."""
        payload = {
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity,
            "count": count,
            "end": str(end),
            "adjust_start_time": 1,
        }
        resp = await self.request(payload, timeout=30)
        return resp.get("candles", [])

    async def get_tick_history(
        self, symbol: str, count: int = 5000, end: str = "latest"
    ) -> list[dict]:
        """Historical ticks, oldest first: [{epoch, price}, ...]."""
        payload = {
            "ticks_history": symbol,
            "style": "ticks",
            "count": count,
            "end": str(end),
            "adjust_start_time": 1,
        }
        resp = await self.request(payload, timeout=30)
        history = resp.get("history") or {}
        return [
            {"epoch": int(t), "price": float(p)}
            for t, p in zip(history.get("times", []), history.get("prices", []))
        ]

    async def get_proposal(
        self,
        contract_type: str,
        symbol: str,
        barrier: str,
        duration: int,
        duration_unit: str,
        stake: float,
        currency: str,
    ) -> dict:
        """Price a vanilla. contract_type: VANILLALONGCALL or VANILLALONGPUT."""
        payload = {
            "proposal": 1,
            "contract_type": contract_type,
            "underlying_symbol": symbol,
            "barrier": barrier,
            "duration": duration,
            "duration_unit": duration_unit,
            "amount": stake,
            "basis": "stake",
            "currency": currency,
        }
        return (await self.request(payload))["proposal"]

    async def get_allowed_barriers(
        self,
        contract_type: str,
        symbol: str,
        duration: int,
        duration_unit: str,
        stake: float,
        currency: str,
    ) -> list[str]:
        """Ask which strike distances Deriv currently allows.

        Vanillas only accept a fixed set of barriers (e.g. +4.40, +2.30, +0.00, ...).
        We send a deliberately odd barrier and read the list out of the error message.
        """
        probe = "+0.01"
        try:
            await self.get_proposal(
                contract_type, symbol, probe, duration, duration_unit, stake, currency
            )
        except DerivAPIError as exc:
            # Short expiries list offsets ("+2.20, -2.20"); long ones list absolute
            # strikes ("1240.00, 1260.00"). Handle both.
            found = re.findall(r"[+-]\d+(?:\.\d+)?", exc.message)
            if not found:
                tail = exc.message.split("available are", 1)[-1]
                found = re.findall(r"\d+(?:\.\d+)?", tail)
            if found:
                return found
            raise
        return [probe]  # the probe itself was valid, so use it

    async def buy(self, proposal_id: str, max_price: float) -> dict:
        """Buy a priced proposal. Never call this until Phase 6 on a real run."""
        return (await self.request({"buy": proposal_id, "price": max_price}))["buy"]