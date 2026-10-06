"""Strategy registry: pick a strategy by name from config."""
from strategies.base import Signal, Strategy
from strategies.breakout import Breakout
from strategies.mean_reversion import MeanReversion
from strategies.trend_pullback import TrendPullback

STRATEGIES = {
    TrendPullback.name: TrendPullback,
    Breakout.name: Breakout,
    MeanReversion.name: MeanReversion,
}


def get_strategy(name: str, **params) -> Strategy:
    try:
        return STRATEGIES[name](**params)
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}. Choose from {sorted(STRATEGIES)}") from None


__all__ = ["Signal", "Strategy", "STRATEGIES", "get_strategy",
           "TrendPullback", "Breakout", "MeanReversion"]