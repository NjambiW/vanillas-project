"""Phase 2 smoke test: live candles plus indicators for about 45 seconds.

Uses 10-second candles so you see results quickly.
Run:  python src/smoke_test_phase2.py
"""
import asyncio
import logging

from config import SYMBOL
from data_feed import CandleFeed
from deriv_client import DerivClient
from indicators import atr, bollinger, ema, rsi

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

GRANULARITY = 10
RUN_SECONDS = 45


def show(frame) -> None:
    close = frame["close"]
    last = frame.iloc[-1]
    print(
        f"{frame.index[-1]:%H:%M:%S}  close {last['close']:.2f}  "
        f"EMA20 {ema(close, 20).iloc[-1]:.2f}  RSI14 {rsi(close, 14).iloc[-1]:.1f}  "
        f"ATR14 {atr(frame, 14).iloc[-1]:.3f}  BBwidth {bollinger(close).iloc[-1]['width']:.3f}"
    )


async def main() -> None:
    client = DerivClient()
    await client.start()
    feed = CandleFeed(client, SYMBOL, GRANULARITY, history_count=200, on_candle=show)
    try:
        await feed.start()
        history = feed.frame()
        print(f"Loaded {len(history)} candles of {GRANULARITY}s. Last 3:")
        print(history.tail(3)[["open", "high", "low", "close"]])
        print(f"\nWatching live for {RUN_SECONDS}s...")
        await asyncio.sleep(RUN_SECONDS)
        await feed.stop()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())