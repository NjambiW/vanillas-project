"""
straddle_test.py - what if every backtest signal bought BOTH call and put?

Reads the CSV written by run_backtest.py (needs roi and roi_opposite columns).
Straddle ROI per 1.0 total staked = (roi + roi_opposite) / 2.

Usage:
    python src/straddle_test.py
    python src/straddle_test.py logs/backtest_20261007_154330.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import LOG_DIR

ROOT = Path(__file__).resolve().parent.parent

N_PERM = 10000
SEED = 1


def find(df, names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def t_stat(x):
    x = np.asarray(x, float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def sign_flip_p(x):
    """One-sided p for 'mean > 0' via random sign flips around zero-centred data
    is not valid here (null is mean = -cost), so we report a t-stat and a bootstrap CI instead."""
    rng = np.random.default_rng(SEED)
    x = np.asarray(x, float)
    boots = [rng.choice(x, len(x), replace=True).mean() for _ in range(N_PERM)]
    return np.percentile(boots, [2.5, 97.5])


def main(path):
    df = pd.read_csv(path)
    roi_c = find(df, ["roi"])
    opp_c = find(df, ["roi_opposite"])
    if roi_c is None or opp_c is None:
        raise SystemExit(f"Need roi and roi_opposite columns. Found: {list(df.columns)}")

    strat_c = find(df, ["strategy", "strategy_name"])
    set_c = find(df, ["set", "split", "dataset"])

    df["straddle"] = (df[roi_c] + df[opp_c]) / 2.0

    group_cols = [c for c in (strat_c, set_c) if c]
    groups = df.groupby(group_cols) if group_cols else [("all", df)]

    print(f"{'group':32s} {'n':>5} {'directional%':>13} {'opposite%':>10} {'straddle%':>10} "
          f"{'t-stat':>7} {'95% CI of straddle %':>24}")
    print("-" * 108)
    for key, g in groups:
        label = " / ".join(map(str, key)) if isinstance(key, tuple) else str(key)
        lo, hi = sign_flip_p(g["straddle"].values)
        print(f"{label:32s} {len(g):>5} {g[roi_c].mean()*100:>13.2f} {g[opp_c].mean()*100:>10.2f} "
              f"{g['straddle'].mean()*100:>10.2f} {t_stat(g['straddle']):>7.2f} "
              f"{f'[{lo*100:.1f}, {hi*100:.1f}]':>24}")

    print("\nHOW TO READ THIS")
    print("  straddle% = average return per 1.0 staked when holding call + put together.")
    print("  If the 95% CI sits entirely below 0, buying volatility at this price loses money.")
    print("  Compare it with the markup (11-14%): a fair-priced straddle would sit near -markup.")
    print("  Expected loss for a straddle bought at fair value x (1+m) is about -m/(1+m).")


def resolve_backtest_csv(arg):
    """Absolute path as-is; relative path tried against cwd, the project root, then logs/."""
    p = Path(arg)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), ROOT, Path(LOG_DIR)):
        if (base / p).exists():
            return base / p
    return ROOT / p


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = resolve_backtest_csv(sys.argv[1])
    else:
        candidates = sorted(Path(LOG_DIR).glob("backtest_*.csv"))
        if not candidates:
            raise SystemExit(
                f"No backtest CSV found in {LOG_DIR} and none given.\n"
                "Run first:  python src/run_backtest.py"
            )
        path = candidates[-1]
        print(f"Using most recent backtest: {path}")
    if not path.exists():
        raise SystemExit(f"Backtest CSV not found: {path}")
    main(path)
