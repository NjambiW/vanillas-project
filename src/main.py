"""The bot. Run:  python src/main.py     (stop with Ctrl+C, or create a file named STOP)

Starts in DRY_RUN mode (see config.py): it watches the market, finds signals, prices them
and logs what it WOULD do, without buying. Set DRY_RUN = False to trade on your demo account.

For a long demo forward test use:  python src/run_demo_forward.py
(it calls main() below with DRY_RUN switched off and a stop event, and prints performance).
"""
import asyncio
import logging
from typing import Optional

from config import (
    CANDLE_SECONDS,
    DRY_RUN,
    FIXED_STAKE,
    HISTORY_COUNT,
    JOURNAL_PATH,
    LOG_DIR,
    MAX_MARKUP_PCT,
    STAKING_PLAN,
    RISK_FRACTION,
    STATUS_EVERY_SECONDS,
    STRATEGY_NAME,
    SYMBOL,
)
from data_feed import CandleFeed
from deriv_client import DerivClient
from executor import Executor
from journal import Journal
from risk import PremiumGate, RiskConfig, RiskManager
from staking import make_plan
from strategies import get_strategy

log = logging.getLogger("bot")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8")],
    )


async def status_loop(client: DerivClient, risk: RiskManager, journal: Journal) -> None:
    while True:
        await asyncio.sleep(STATUS_EVERY_SECONDS)
        try:
            bal = await client.get_balance()
            s = journal.summary()
            log.info(
                "STATUS balance %s %s | pnl today %.2f | closed %d (win rate %s) | total %.2f",
                bal["balance"], bal["currency"], risk.pnl_today, s["trades_closed"],
                f"{s['win_rate']:.0%}" if s["win_rate"] is not None else "n/a", s["total_profit"],
            )
        except Exception:  # noqa: BLE001
            log.exception("status check failed")


async def main(stop_event: Optional[asyncio.Event] = None) -> None:
    """Run the bot until Ctrl+C, or until `stop_event` is set (if one is given)."""
    setup_logging()
    strategy = get_strategy(STRATEGY_NAME)
    plan_params = ({"fraction": RISK_FRACTION} if STAKING_PLAN == "fixed_fraction"
                   else {"amount": FIXED_STAKE} if STAKING_PLAN == "fixed" else {})
    risk = RiskManager(RiskConfig.from_settings(), make_plan(STAKING_PLAN, **plan_params),
                       state_path=LOG_DIR / "risk_state.json")
    gate = PremiumGate(MAX_MARKUP_PCT)
    journal = Journal(JOURNAL_PATH)

    client = DerivClient()
    await client.start()
    executor = Executor(
        client, strategy, risk, gate, journal, SYMBOL, CANDLE_SECONDS, dry_run=DRY_RUN,
        proposal_log=LOG_DIR / "proposals.csv",
    )

    loop = asyncio.get_running_loop()
    pending: set = set()

    def _finished(task: asyncio.Task) -> None:
        pending.discard(task)
        if not task.cancelled() and task.exception() is not None:
            # create_task drops the result, so without this the traceback is never shown
            log.error("candle handler failed", exc_info=task.exception())

    def on_candle(frame) -> None:
        task = loop.create_task(executor.handle_candle(frame))
        pending.add(task)
        task.add_done_callback(_finished)

    feed = CandleFeed(client, SYMBOL, CANDLE_SECONDS, HISTORY_COUNT, on_candle=on_candle)
    await feed.start()
    log.info(
        "Bot running: %s on %s, %ss candles, %s mode. Ctrl+C to stop.",
        strategy.name, SYMBOL, CANDLE_SECONDS, "DRY RUN" if DRY_RUN else "DEMO TRADING",
    )
    status = asyncio.create_task(status_loop(client, risk, journal))
    status.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    try:
        await (stop_event or asyncio.Event()).wait()  # run until interrupted or told to stop
    finally:
        status.cancel()
        await feed.stop()
        if executor._tasks:
            log.info("waiting for open contracts to settle (Ctrl+C again to force quit)")
            await executor.wait_idle()
        await client.close()
        s = journal.summary()
        log.info("FINAL SUMMARY %s", s)
        journal.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")