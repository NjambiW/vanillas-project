"""Stake sizing plans.

A plan answers one question: "how much should the next trade stake?"
Risk limits (risk.py) are applied on top, so a plan can never exceed them.

Honest note: no staking plan changes whether a strategy makes money. They only change
how wins and losses are spread out. The edge has to come from the strategy.
"""
import math
from abc import ABC, abstractmethod


def floor_cents(amount: float) -> float:
    """Round DOWN to 2 decimals so rounding can never push a stake over a limit."""
    return math.floor(amount * 100 + 1e-9) / 100


class StakingPlan(ABC):
    name = "base"

    @abstractmethod
    def next_stake(self, balance: float) -> float:
        """Desired stake for the next trade, before risk limits."""

    def record(self, profit: float) -> None:
        """Called after each settled trade with its profit (negative for a loss)."""


class FixedStake(StakingPlan):
    name = "fixed"

    def __init__(self, amount: float = 1.0):
        self.amount = amount

    def next_stake(self, balance: float) -> float:
        return floor_cents(self.amount)


class FixedFraction(StakingPlan):
    """Stake a fixed share of the current balance. Stakes shrink after losses."""

    name = "fixed_fraction"

    def __init__(self, fraction: float = 0.01):
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be between 0 and 1")
        self.fraction = fraction

    def next_stake(self, balance: float) -> float:
        return floor_cents(balance * self.fraction)


class OscarsGrind(StakingPlan):
    """
    Aims to finish each cycle one unit up. Stake stays the same after a loss and goes up
    one unit after a win, until the cycle is one unit in profit, then it resets.
    Vanilla payouts vary, so this is an adaptation of the classic fixed-odds version.
    """

    name = "oscars_grind"

    def __init__(self, unit: float = 1.0, max_units: int = 5):
        self.unit = unit
        self.max_units = max_units
        self._units = 1
        self._cycle_profit = 0.0

    def next_stake(self, balance: float) -> float:
        return floor_cents(self.unit * self._units)

    def record(self, profit: float) -> None:
        self._cycle_profit += profit
        if self._cycle_profit >= self.unit:
            self._units, self._cycle_profit = 1, 0.0
        elif profit > 0:
            self._units = min(self._units + 1, self.max_units)


class AntiMartingale(StakingPlan):
    """Press winners, cut losers: stake grows after each win and resets after a loss."""

    name = "anti_martingale"

    def __init__(self, base: float = 1.0, factor: float = 1.5, max_steps: int = 3):
        self.base, self.factor, self.max_steps = base, factor, max_steps
        self._step = 0

    def next_stake(self, balance: float) -> float:
        return floor_cents(self.base * self.factor**self._step)

    def record(self, profit: float) -> None:
        self._step = min(self._step + 1, self.max_steps) if profit > 0 else 0


PLANS = {cls.name: cls for cls in (FixedStake, FixedFraction, OscarsGrind, AntiMartingale)}


def make_plan(name: str, **params) -> StakingPlan:
    try:
        return PLANS[name](**params)
    except KeyError:
        raise ValueError(f"unknown staking plan {name!r}. Choose from {sorted(PLANS)}") from None