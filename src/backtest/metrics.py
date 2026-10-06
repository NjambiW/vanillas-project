"""Turn a list of backtest trades into the numbers that matter."""
import math

import numpy as np


def summarize(trades: list, balances: list) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    profits = np.array([t.profit for t in trades])
    stakes = np.array([t.stake for t in trades])
    wins, losses = profits[profits > 0], profits[profits <= 0]

    peak, max_dd, max_dd_pct = balances[0], 0.0, 0.0
    for b in balances:
        peak = max(peak, b)
        max_dd = max(max_dd, peak - b)
        max_dd_pct = max(max_dd_pct, (peak - b) / peak * 100 if peak > 0 else 0.0)

    streak = longest = 0
    for p in profits:
        streak = streak + 1 if p <= 0 else 0
        longest = max(longest, streak)

    fair_costs = np.array([t.fair_cost for t in trades])
    payoffs = np.array([t.payoff for t in trades])
    return {
        "trades": n,
        "win_rate": float((profits > 0).mean()),
        "total_profit": float(profits.sum()),
        "roi": float((profits / stakes).mean()),                   # after Deriv's markup
        "gross_roi": float((payoffs / fair_costs - 1).mean()),     # before the markup
        "avg_markup_pct": float(np.mean([t.markup_pct for t in trades])),
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else math.inf,
        "max_drawdown": float(max_dd),
        "max_drawdown_pct": float(max_dd_pct),
        "longest_losing_streak": longest,
        "final_balance": float(balances[-1]),
    }


def direction_test(trades: list, shuffles: int = 2000, seed: int = 0) -> dict:
    """Is the strategy's direction better than a coin flip at the same entry times?

    For each trade we know the result of the direction it chose and of the opposite one.
    We replay the same entries with random directions many times. The p-value is the share
    of random replays that did at least as well. Small = the direction carries real information.
    """
    if not trades:
        return {"p_value": None}
    chosen = np.array([t.roi for t in trades])
    opposite = np.array([t.roi_opposite for t in trades])
    rng = np.random.default_rng(seed)
    flips = rng.random((shuffles, len(trades))) < 0.5
    random_means = np.where(flips, opposite, chosen).mean(axis=1)
    observed = float(chosen.mean())
    p = (int((random_means >= observed).sum()) + 1) / (shuffles + 1)
    return {
        "observed_roi": observed,
        "random_direction_roi": float(random_means.mean()),
        "edge_over_random": observed - float(random_means.mean()),
        "p_value": p,
    }
