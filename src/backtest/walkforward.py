"""Honest out-of-sample evaluation: purged walk-forward folds and block-bootstrap CIs.

Two problems with the original report:

  * a single train/test split says nothing about stability across time. Once several
    strategies, holds, strikes and volatility buckets have been tried on the test
    segment, that segment has taken part in model selection and is validation data,
    not a test set;
  * a p-value that shuffles individual trade directions treats overlapping contracts
    as independent observations. A five-candle contract entered on consecutive candles
    shares most of its future with its neighbour, so 1,199 trades are not 1,199
    independent samples.

So: expanding-window walk-forward folds with a purge gap of at least the maximum
holding period between train and test, and a moving-block bootstrap confidence
interval over chronological trade returns. A strategy passes only when the combined
out-of-sample CI lower bound is above zero, not merely because its mean is positive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def purged_walk_forward(
    n_rows: int,
    train_size: int,
    test_size: int,
    purge: int,
    step: int | None = None,
) -> Iterator[Fold]:
    """Expanding-window walk-forward folds.

    `train_end`, `test_start` and `test_end` are Python-exclusive indexes. The purge
    keeps `purge` rows between the end of training and the start of testing so a
    training label cannot extend into the test period; pass at least the maximum
    holding period.
    """
    if min(n_rows, train_size, test_size) <= 0:
        raise ValueError("sizes must be positive")
    if purge < 0:
        raise ValueError("purge cannot be negative")

    step = test_size if step is None else step
    test_start = train_size + purge

    while test_start + test_size <= n_rows:
        yield Fold(
            train_start=0,
            train_end=test_start - purge,
            test_start=test_start,
            test_end=test_start + test_size,
        )
        test_start += step


def block_bootstrap_mean_ci(
    returns: "np.ndarray | pd.Series",
    blocksize: int,
    simulations: int = 10000,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Moving-block bootstrap 95% CI for the mean of a chronological return series.

    Use `blocksize >=` the maximum holding period, and preferably estimate it
    conservatively when trades overlap.
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan, np.nan, np.nan
    if blocksize < 1:
        raise ValueError("blocksize must be at least 1")

    blocksize = min(blocksize, len(x))
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(x) - blocksize + 1)
    blocks_needed = int(np.ceil(len(x) / blocksize))
    means = np.empty(simulations)

    for i in range(simulations):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([x[start:start + blocksize] for start in chosen])[:len(x)]
        means[i] = sample.mean()

    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(x.mean()), float(lower), float(upper)


def blocksize_for(hold: int, n_trades: int) -> int:
    """Bootstrap block length: at least the holding period, capped so the CI stays honest.

    The guidance is `blocksize >=` the maximum holding period, but if the sample is
    short that leaves a single block and a zero-width, falsely precise interval. Capping
    at a quarter of the sample guarantees at least four blocks are drawn.
    """
    if n_trades < 2:
        return 1
    return max(1, min(hold, n_trades // 4))


def summarize_fold_trades(
    trades: pd.DataFrame,
    returncolumn: str = "netreturn",
    blocksize: int = 5,
) -> dict:
    """Mean net ROI and its block-bootstrap CI for one fold's trades."""
    if trades.empty:
        return {
            "trades": 0,
            "mean": np.nan,
            "cilow": np.nan,
            "cihigh": np.nan,
            "profitable": False,
        }

    mean, low, high = block_bootstrap_mean_ci(
        trades[returncolumn],
        blocksize=blocksize,
    )
    return {
        "trades": len(trades),
        "mean": mean,
        "cilow": low,
        "cihigh": high,
        "profitable": bool(low > 0.0),
    }
