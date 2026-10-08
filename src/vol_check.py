"""
vol_check.py - does realised-vs-long-run volatility predict straddle profit?

Idea: on a constant-vol index the option price (fair value x (1+markup)) is
based on ONE volatility. If short-term realised vol sometimes runs above that,
buying straddles only then could beat the cost. This script checks that.

Usage:
    python src/vol_check.py
    python src/vol_check.py --candles data/1HZ100V_60s.csv --hold 5 --markup 0.12
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from config import CANDLE_SECONDS, DATA_DIR, LOG_DIR, SYMBOL

# ---------------- config ----------------
CLOSE_COL_CANDIDATES = ["close", "Close", "c", "price"]
SHORT_WINDOW = 30        # candles for "recent" realised vol
LONG_WINDOW = 1440       # candles for "normal" realised vol
TRAIN_FRAC = 0.7
N_PERM = 5000
SEED = 42
# ----------------------------------------

_erf = np.vectorize(math.erf)


def norm_cdf(x):
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def bs_price(S, K, sig, T, kind):
    """Black-Scholes, r=0. sig is per-sqrt(candle), T is in candles."""
    sd = sig * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sd ** 2) / sd
    d2 = d1 - sd
    if kind == "call":
        return S * norm_cdf(d1) - K * norm_cdf(d2)
    return K * norm_cdf(-d2) - S * norm_cdf(-d1)


def load_close(path):
    df = pd.read_csv(path)
    for c in CLOSE_COL_CANDIDATES:
        if c in df.columns:
            return df[c].astype(float).to_numpy()
    raise SystemExit(f"No close column found. Columns: {list(df.columns)}")


def perm_pvalue(all_roi, sel_mask, rng):
    """P(random subset of same size has mean ROI >= selected subset)."""
    n = int(sel_mask.sum())
    if n < 5:
        return float("nan")
    obs = all_roi[sel_mask].mean()
    cnt = 0
    for _ in range(N_PERM):
        idx = rng.choice(len(all_roi), n, replace=False)
        if all_roi[idx].mean() >= obs:
            cnt += 1
    return (cnt + 1) / (N_PERM + 1)


def run(args):
    close = load_close(args.candles)
    rets = np.diff(np.log(close))
    n = len(rets)
    split = int(n * TRAIN_FRAC)

    # constant "pricing" vol = train-set vol (what a flat-vol index would quote)
    sig_price = rets[:split].std()
    print(f"Candles: {len(close)}  train/test split at return {split}")
    print(f"Pricing vol per candle (train): {sig_price:.6f}")

    rows = []
    h = args.hold
    start = max(LONG_WINDOW, SHORT_WINDOW) + 1
    for t in range(start, n - h, h):          # non-overlapping entries
        s_short = rets[t - SHORT_WINDOW:t].std()
        s_long = rets[t - LONG_WINDOW:t].std()
        ratio = s_short / s_long if s_long > 0 else np.nan
        S0 = close[t]                          # price at entry (after return t-1)
        ST = close[t + h]
        d = args.strike_mult * sig_price * S0 * math.sqrt(h)
        rows.append((t, ratio, S0, ST, d))

    df = pd.DataFrame(rows, columns=["t", "ratio", "S0", "ST", "d"])
    S0, ST, d = df.S0.values, df.ST.values, df.d.values

    Kc, Kp = S0 + d, S0 - d
    call_fv = bs_price(S0, Kc, sig_price, h, "call")
    put_fv = bs_price(S0, Kp, sig_price, h, "put")
    call_pay = np.maximum(ST - Kc, 0)
    put_pay = np.maximum(Kp - ST, 0)

    for label, mk in [("GROSS (no markup)", 0.0), (f"NET (markup {args.markup:.1%})", args.markup)]:
        prem = (call_fv + put_fv) * (1 + mk)
        df["roi_" + ("gross" if mk == 0 else "net")] = (call_pay + put_pay) / prem - 1

    df["is_train"] = df.t < split
    rng = np.random.default_rng(SEED)

    print(f"\nStraddle/strangle: strike offset {args.strike_mult} x sigma*sqrt(hold), hold {h} candles")
    print("-" * 78)
    for name, sub in [("train", df[df.is_train]), ("test", df[~df.is_train])]:
        print(f"{name:5s} trades {len(sub):5d}  gross ROI {sub.roi_gross.mean()*100:7.2f}%  "
              f"net ROI {sub.roi_net.mean()*100:7.2f}%")

    # --- bucket by vol ratio (cut points from TRAIN so test is untouched) ---
    cuts = np.nanquantile(df[df.is_train].ratio, [0.2, 0.4, 0.6, 0.8])
    df["bucket"] = np.digitize(df.ratio, cuts)
    print("\nNet ROI by recent-vol ratio bucket (0 = calmest, 4 = most volatile):")
    print(f"{'bucket':>6} {'ratio range':>20} {'train n':>8} {'train ROI%':>11} {'test n':>7} {'test ROI%':>10}")
    edges = [-np.inf, *cuts, np.inf]
    for b in range(5):
        tr = df[(df.bucket == b) & df.is_train]
        te = df[(df.bucket == b) & ~df.is_train]
        rng_txt = f"{edges[b]:.2f} to {edges[b+1]:.2f}"
        print(f"{b:>6} {rng_txt:>20} {len(tr):>8} {tr.roi_net.mean()*100:>11.2f} "
              f"{len(te):>7} {te.roi_net.mean()*100:>10.2f}")

    # --- pre-specified rule: trade only the top bucket ---
    test = df[~df.is_train].reset_index(drop=True)
    mask = (test.bucket == 4).values
    p = perm_pvalue(test.roi_net.values, mask, rng)
    print(f"\nRule 'trade only when ratio > {cuts[-1]:.2f}' on TEST: "
          f"n={mask.sum()}, net ROI {test.roi_net[mask].mean()*100:.2f}% "
          f"(all test entries: {test.roi_net.mean()*100:.2f}%), permutation p={p:.3f}")

    # sanity: does recent vol predict FUTURE realised vol at all?
    fut = np.abs(np.log(df.ST / df.S0))
    corr = np.corrcoef(df.ratio.fillna(1), fut)[0, 1]
    print(f"Correlation of recent-vol ratio with next |move|: {corr:.3f} "
          f"(near 0 = volatility is not predictable)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved per-trade results to {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candles", default=str(DATA_DIR / f"{SYMBOL}_{CANDLE_SECONDS}s.csv"),
                    help="CSV with a close column (60s candles); defaults to the downloaded history")
    ap.add_argument("--hold", type=int, default=5, help="hold time in candles")
    ap.add_argument("--markup", type=float, default=0.12, help="measured markup, e.g. 0.12")
    ap.add_argument("--strike_mult", type=float, default=0.0,
                    help="0 = straddle; >0 = strangle, strikes this many sigma*sqrt(hold) away")
    ap.add_argument("--out", default=str(LOG_DIR / "vol_check_trades.csv"))
    run(ap.parse_args())
