"""Download historical candles to a CSV file for backtesting.

Examples:
    python src/download_history.py
    python src/download_history.py --symbol 1HZ100V --granularity 60 --total 50000

The window always ends at the latest candle, so every download moves forward in time.
New candles are therefore MERGED with whatever is already on disk rather than
replacing it: an overwrite would quietly drop the oldest days, which are the ones the
out-of-sample test needs.
"""
import argparse
import asyncio
import logging

import pandas as pd

from config import CANDLE_SECONDS, DATA_DIR, SYMBOL
from data_feed import (CANDLE_COLUMNS, STANDARD_GRANULARITIES, candles_to_frame,
                       load_candles_csv, save_candles_csv)
from deriv_client import DerivClient

PAGE_SIZE = 5000

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


async def download(symbol: str, granularity: int, total: int):
    client = DerivClient()
    await client.start()
    rows: list[dict] = []
    end = "latest"
    try:
        while len(rows) < total:
            batch = await client.get_candles(
                symbol, granularity, min(PAGE_SIZE, total - len(rows)), end
            )
            if not batch:
                print("No more history available.")
                break
            earliest = min(int(c["epoch"]) for c in batch)
            rows = batch + rows
            print(f"Got {len(batch)} candles, total {len(rows)}, earliest epoch {earliest}")
            if end != "latest" and earliest >= int(end) + 1:
                print("No older data returned, stopping.")
                break
            end = str(earliest - 1)
            await asyncio.sleep(0.3)
    finally:
        await client.close()
    return candles_to_frame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--granularity", type=int, default=CANDLE_SECONDS)
    parser.add_argument("--total", type=int, default=20000)
    args = parser.parse_args()
    if args.granularity not in STANDARD_GRANULARITIES:
        parser.error(f"granularity must be one of {sorted(STANDARD_GRANULARITIES)}")

    df = asyncio.run(download(args.symbol, args.granularity, args.total))
    path = DATA_DIR / f"{args.symbol}_{args.granularity}s.csv"
    if df.empty:
        # A failed download must not truncate the history we already have.
        raise SystemExit(f"No candles returned; {path} left untouched.")
    if path.exists():
        on_disk = load_candles_csv(path)
        if len(on_disk):
            merged = candles_to_frame(pd.concat(
                [on_disk[CANDLE_COLUMNS], df[CANDLE_COLUMNS]], ignore_index=True
            ).to_dict("records"))
            print(f"Kept {len(on_disk)} candles already on disk "
                  f"(new: {len(df)}, after de-duplication: {len(merged)}).")
            df = merged
    save_candles_csv(df, path)
    print(f"Saved {len(df)} candles to {path}")
    if len(df):
        print(f"From {df.index[0]} to {df.index[-1]}")


if __name__ == "__main__":
    main()