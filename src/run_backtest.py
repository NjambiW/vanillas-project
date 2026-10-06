"""Phase 7: backtest the strategies on your downloaded history.

Needs:  python src/download_history.py   (candles)   and ideally   python src/measure_markup.py
Examples:
    python src/run_backtest.py
    python src/run_backtest.py --strategy breakout --hold 5 15 60
    python src/run_backtest.py --plan fixed_fraction --balance 1000

Each strategy is tested on the first part of the data (train) and then on the last part
(test) that it never saw. A result only counts if it holds up on BOTH.
"""
import argparse
from datetime import datetime

import pandas as pd

from backtest.cost_model import CostModel, MEASURED, latest_markup_csv, load_markup_csv
from backtest.engine import run_backtest
from backtest.metrics import direction_test, summarize
from config import (CANDLE_SECONDS, DATA_DIR, LOG_DIR, MAX_RISK_PCT_PER_TRADE, MAX_STAKE,
                    MIN_STAKE, RISK_FRACTION, SYMBOL)
from data_feed import load_candles_csv
from staking import make_plan
from strategies import STRATEGIES, get_strategy


def build_cost_model(path_arg):
    path = path_arg or latest_markup_csv(LOG_DIR)
    table = dict(MEASURED)
    source = "built-in table (measured 2026-10-06)"
    if path:
        measured = load_markup_csv(path, SYMBOL)
        if measured:
            table.update(measured)
            source = f"{path} (+ built-in table for missing expiries)"
    return CostModel(table), source


def pct(x):
    return "   n/a" if x is None else f"{x * 100:6.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="all", help="all or one of " + ", ".join(STRATEGIES))
    ap.add_argument("--hold", type=int, nargs="+", help="expiry in candles; several values = a sweep")
    ap.add_argument("--strike-atr", type=float, default=None, help="strike distance in ATRs")
    ap.add_argument("--split", type=float, default=0.7, help="share of data used as train")
    ap.add_argument("--plan", default="fixed", help="fixed or fixed_fraction")
    ap.add_argument("--stake", type=float, default=1.0, help="stake for the fixed plan")
    ap.add_argument("--balance", type=float, default=1000.0)
    ap.add_argument("--shuffles", type=int, default=2000)
    ap.add_argument("--markup-csv", default=None)
    args = ap.parse_args()

    path = DATA_DIR / f"{SYMBOL}_{CANDLE_SECONDS}s.csv"
    if not path.exists():
        raise SystemExit(f"No data at {path}. Run: python src/download_history.py")
    candles = load_candles_csv(path)
    cut = int(len(candles) * args.split)
    segments = {"train": candles.iloc[:cut].reset_index(drop=True),
                "test": candles.iloc[cut:].reset_index(drop=True)}
    cost_model, source = build_cost_model(args.markup_csv)

    print(f"{SYMBOL}: {len(candles)} candles of {CANDLE_SECONDS}s, "
          f"train {len(segments['train'])} / test {len(segments['test'])}")
    print(f"Prices modelled as fair value x (1 + measured markup). Markup source: {source}\n")

    names = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    header = (f"{'strategy':15}{'hold':>5}{'set':>6}{'trades':>7}{'win%':>6}{'ROI%':>7}"
              f"{'grossROI%':>10}{'markup%':>8}{'maxDD%':>7}{'p-value':>8}")
    print(header)
    print("-" * len(header))

    all_rows, summaries, tests_run = [], [], 0
    for name in names:
        for hold in (args.hold or [None]):
            params = {}
            if hold is not None:
                params["hold_candles"] = hold
            if args.strike_atr is not None:
                params["strike_atr"] = args.strike_atr
            outcome = {}
            for seg_name, frame in segments.items():
                strat = get_strategy(name, **params)
                plan = (make_plan("fixed", amount=args.stake) if args.plan == "fixed"
                        else make_plan("fixed_fraction", fraction=RISK_FRACTION))
                res = run_backtest(frame, strat, SYMBOL, cost_model, plan, args.balance,
                                   MIN_STAKE if args.plan != "fixed" else 0.4, MAX_STAKE,
                                   MAX_RISK_PCT_PER_TRADE)
                m = summarize(res.trades, res.balances)
                if m["trades"] == 0:
                    print(f"{name:15}{strat.hold_candles:>5}{seg_name:>6}{0:>7}   (no trades)")
                    outcome[seg_name] = None
                    continue
                t = direction_test(res.trades, args.shuffles)
                tests_run += 1
                print(f"{name:15}{strat.hold_candles:>5}{seg_name:>6}{m['trades']:>7}"
                      f"{m['win_rate'] * 100:6.1f}{pct(m['roi']):>7}{pct(m['gross_roi']):>10}"
                      f"{m['avg_markup_pct']:8.1f}{m['max_drawdown_pct']:7.1f}{t['p_value']:8.3f}")
                outcome[seg_name] = (m, t)
                for tr in res.trades:
                    all_rows.append({"strategy": name, "hold": strat.hold_candles, "set": seg_name, **tr.__dict__})
            summaries.append((name, params.get("hold_candles", get_strategy(name).hold_candles), outcome))

    candidates = []
    for name, hold, outcome in summaries:
        tr, te = outcome.get("train"), outcome.get("test")
        if tr and te and tr[0]["roi"] > 0 and te[0]["roi"] > 0 and tr[1]["p_value"] < 0.05 and te[1]["p_value"] < 0.05:
            candidates.append((name, hold))

    print("\nHOW TO READ THIS")
    print("  ROI% is the average return per 1.0 staked AFTER Deriv's markup; grossROI% is before it.")
    print("  p-value: chance that random directions at the same entries would have done as well.")
    print(f"  You ran {tests_run} tests, so a few p-values below 0.05 are expected by luck alone.")
    if candidates:
        print(f"\nPassed on train AND test (ROI > 0, p < 0.05): {candidates}")
        print("  Treat these as leads to re-test on fresh data, not as proven edges.")
    else:
        print("\nNo strategy was profitable after costs with a real edge on both train and test.")

    if all_rows:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = LOG_DIR / f"backtest_{datetime.now():%Y%m%d_%H%M%S}.csv"
        pd.DataFrame(all_rows).to_csv(out, index=False)
        print(f"\nAll trades saved to {out}")


if __name__ == "__main__":
    main()