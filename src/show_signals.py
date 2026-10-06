"""Run every strategy over the downloaded candle history and summarise the signals.

Needs the CSV from:  python src/download_history.py
Run:  python src/show_signals.py
"""
import sys

from config import CANDLE_SECONDS, DATA_DIR, SYMBOL
from data_feed import load_candles_csv
from strategies import STRATEGIES, get_strategy


def main() -> None:
    path = DATA_DIR / f"{SYMBOL}_{CANDLE_SECONDS}s.csv"
    if not path.exists():
        print(f"No data file at {path}.\nRun first:  python src/download_history.py")
        sys.exit(1)

    candles = load_candles_csv(path)
    print(f"{len(candles)} candles, {candles.index[0]} to {candles.index[-1]}\n")

    for name in STRATEGIES:
        strat = get_strategy(name)
        sig = strat.signal_series(candles)
        calls, puts = int((sig > 0).sum()), int((sig < 0).sum())
        per_100 = (calls + puts) / len(candles) * 100
        print(f"{name:16} CALL {calls:5}  PUT {puts:5}  ({per_100:.1f} signals per 100 candles)")
        for when, value in sig[sig != 0].tail(3).items():
            print(f"    {when:%Y-%m-%d %H:%M}  {'CALL' if value > 0 else 'PUT'}")


if __name__ == "__main__":
    main()