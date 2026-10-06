"""Turns a strategy Signal into (at most) one Deriv vanilla trade, and follows it to settlement.

Flow for each signal:
    risk approval -> pick expiry and strike -> get a fresh proposal -> price gate
    -> buy (unless DRY_RUN) -> watch the contract until it settles -> record the result
Every refusal is written to the journal with its reason.
"""
import asyncio
import logging
from typing import Optional

import pandas as pd

from deriv_client import DerivAPIError, DerivClient
from journal import Journal
from markup import ASK_PRICE_KEYS, PAYOUT_PER_POINT_KEYS, find_field
from risk import PremiumGate, RiskManager
from strategies.base import Signal, Strategy

log = logging.getLogger("executor")

MAX_CONSECUTIVE_ERRORS = 3


def seconds_to_duration(seconds: int) -> tuple:
    """(value, unit) in the largest whole unit: 3600 -> (1, 'h'), 300 -> (5, 'm')."""
    seconds = int(seconds)
    if seconds % 86400 == 0:
        return seconds // 86400, "d"
    if seconds % 3600 == 0:
        return seconds // 3600, "h"
    if seconds % 60 == 0:
        return seconds // 60, "m"
    return seconds, "s"


def barrier_offset(barrier: str, spot: float) -> float:
    """Offset from spot in points. '+2.20' is already an offset; '1240.00' is an absolute strike."""
    value = float(barrier)
    return value if barrier.strip().startswith(("+", "-")) else value - spot


def pick_barrier(allowed: list, wanted_offset: float, spot: float = 0.0) -> str:
    """The allowed barrier closest to the wanted offset (+ above spot, - below)."""
    return min(allowed, key=lambda b: abs(barrier_offset(b, spot) - wanted_offset))


def _truthy(value) -> bool:
    return value in (1, True, "1", "true", "True")


class Executor:
    def __init__(
        self,
        client: DerivClient,
        strategy: Strategy,
        risk: RiskManager,
        gate: PremiumGate,
        journal: Journal,
        symbol: str,
        granularity: int,
        dry_run: bool = True,
        settle_buffer: float = 120.0,
    ):
        self.client, self.strategy, self.risk = client, strategy, risk
        self.gate, self.journal = gate, journal
        self.symbol, self.granularity = symbol, granularity
        self.dry_run = dry_run
        self.settle_buffer = settle_buffer
        self._lock = asyncio.Lock()
        self._tasks: set = set()
        self._errors = 0

    # ------------------------------------------------------------- entry points
    async def handle_candle(self, candles: pd.DataFrame) -> None:
        """Call this each time a candle closes."""
        signal = self.strategy.evaluate(candles)
        if signal is None:
            return
        price = float(candles["close"].iloc[-1])
        self.journal.log_signal(self.strategy.name, signal.direction, signal.reason, signal.atr, price)
        async with self._lock:
            try:
                await self.handle_signal(signal, price)
                self._errors = 0
            except Exception:  # noqa: BLE001  never let one bad trade attempt kill the bot
                log.exception("error while handling signal")
                self._errors += 1
                if self._errors >= MAX_CONSECUTIVE_ERRORS:
                    self.risk.halt(f"{self._errors} errors in a row, check logs/bot.log")

    async def wait_idle(self) -> None:
        """Wait for all open-contract watchers to finish (used by tests and shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # --------------------------------------------------------------- the trade
    async def handle_signal(self, signal: Signal, price: float) -> Optional[int]:
        name = self.strategy.name

        def skip(reason: str) -> None:
            log.info("skip %s: %s", signal.direction, reason)
            self.journal.log_skip(name, signal.direction, reason)

        bal = await self.client.get_balance()
        balance, currency = float(bal["balance"]), bal["currency"]

        decision = self.risk.approve_trade(balance)
        if not decision.allowed:
            skip(decision.reason)
            return None

        duration, unit = seconds_to_duration(signal.hold_candles * self.granularity)
        allowed = await self.client.get_allowed_barriers(
            signal.contract_type, self.symbol, duration, unit, decision.stake, currency
        )
        offset = signal.strike_distance if signal.direction == "CALL" else -signal.strike_distance
        barrier = pick_barrier(allowed, offset, price)

        proposal = await self.client.get_proposal(
            signal.contract_type, self.symbol, barrier, duration, unit, decision.stake, currency
        )
        stake = decision.stake
        min_stake, max_stake = proposal.get("min_stake"), proposal.get("max_stake")
        if min_stake is not None and stake < float(min_stake):
            skip(f"stake {stake} is below Deriv's minimum {min_stake}")
            return None
        if max_stake is not None and stake > float(max_stake):
            skip(f"stake {stake} is above Deriv's maximum {max_stake}")
            return None

        verdict = self.gate.check(
            signal.contract_type, proposal, price, barrier, duration, unit, self.symbol
        )
        if not verdict.allowed:
            skip(verdict.reason)
            return None

        if self.dry_run:
            skip(
                f"DRY RUN: would buy {signal.contract_type} {duration}{unit} barrier {barrier} "
                f"stake {stake} (markup {verdict.markup_pct:.1f}%)"
            )
            return None

        try:
            bought = await self.client.buy(proposal["id"], float(find_field(proposal, ASK_PRICE_KEYS)))
        except DerivAPIError as exc:
            skip(f"buy rejected: {exc}")
            return None
        except asyncio.TimeoutError:
            self.risk.halt("buy response timed out; check the account's open positions before resuming")
            skip("buy timed out, bot halted")
            return None

        contract_id = bought.get("contract_id")
        self.risk.on_trade_opened(stake)
        trade_id = self.journal.open_trade(
            strategy=name,
            contract_id=contract_id,
            contract_type=signal.contract_type,
            symbol=self.symbol,
            duration=f"{duration}{unit}",
            barrier=barrier,
            strike=_to_float((proposal.get("contract_details") or {}).get("barrier")),
            spot=_to_float(proposal.get("spot")),
            stake=stake,
            payout_per_point=_to_float(find_field(proposal, PAYOUT_PER_POINT_KEYS)),
            markup_pct=verdict.markup_pct,
        )
        log.info("BOUGHT %s %s%s barrier %s stake %s (contract %s)",
                 signal.contract_type, duration, unit, barrier, stake, contract_id)

        expiry_seconds = signal.hold_candles * self.granularity
        task = asyncio.create_task(self._watch(contract_id, trade_id, expiry_seconds))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return trade_id

    # ------------------------------------------------------- follow to settlement
    async def _watch(self, contract_id, trade_id: int, expiry_seconds: float) -> None:
        loop = asyncio.get_running_loop()
        done: asyncio.Future = loop.create_future()

        def on_update(msg: dict) -> None:
            poc = msg.get("proposal_open_contract") or {}
            if (_truthy(poc.get("is_sold")) or _truthy(poc.get("is_expired"))) and not done.done():
                done.set_result(poc)

        sub_id = None
        try:
            first = await self.client.subscribe(
                {"proposal_open_contract": 1, "contract_id": contract_id}, on_update
            )
            sub_id = (first.get("subscription") or {}).get("id")
            poc = await asyncio.wait_for(done, expiry_seconds + self.settle_buffer)
        except Exception as exc:  # noqa: BLE001  timeout, API error, dropped connection, anything
            poc = await self._last_chance_check(contract_id, exc)
        finally:
            if sub_id:
                try:
                    await self.client.forget(sub_id)
                except Exception:  # noqa: BLE001
                    log.warning("could not release subscription %s", sub_id)

        if poc is None:
            self.journal.close_trade(trade_id, None, "unknown")
            self.risk.release_position()
            self.risk.halt(f"could not confirm settlement of contract {contract_id}; check the account")
            log.error("settlement of contract %s unknown, bot halted", contract_id)
            return

        profit = float(poc.get("profit", 0))
        self.journal.close_trade(trade_id, profit, "won" if profit > 0 else "lost")
        self.risk.on_trade_closed(profit)
        log.info("SETTLED contract %s profit %.2f", contract_id, profit)

    async def _last_chance_check(self, contract_id, original_error) -> Optional[dict]:
        """One direct lookup before giving up on a contract."""
        log.warning("watching contract %s failed (%r), checking directly", contract_id, original_error)
        try:
            resp = await self.client.request({"proposal_open_contract": 1, "contract_id": contract_id})
        except Exception:  # noqa: BLE001
            return None
        poc = resp.get("proposal_open_contract") or {}
        return poc if (_truthy(poc.get("is_sold")) or _truthy(poc.get("is_expired"))) else None


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None