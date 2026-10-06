import pytest

from staking import AntiMartingale, FixedFraction, FixedStake, OscarsGrind, floor_cents, make_plan


def test_floor_cents_never_rounds_up():
    assert floor_cents(1.239) == 1.23
    assert floor_cents(10.0) == 10.0
    assert floor_cents(0.999) == 0.99


def test_fixed_stake():
    assert FixedStake(2.5).next_stake(10_000) == 2.5


def test_fixed_fraction_scales_with_balance():
    plan = FixedFraction(0.01)
    assert plan.next_stake(1000) == 10.0
    assert plan.next_stake(900) == 9.0
    with pytest.raises(ValueError):
        FixedFraction(0)


def test_oscars_grind_cycle():
    plan = OscarsGrind(unit=1.0, max_units=5)
    assert plan.next_stake(0) == 1.0
    plan.record(0.8)            # win, but cycle still below +1 unit: stake up one unit
    assert plan.next_stake(0) == 2.0
    plan.record(-2.0)           # loss: stake unchanged
    assert plan.next_stake(0) == 2.0
    plan.record(3.0)            # cycle now +1.8, target reached: reset
    assert plan.next_stake(0) == 1.0


def test_oscars_grind_is_capped():
    plan = OscarsGrind(unit=1.0, max_units=3)
    plan.record(-5.0)           # put the cycle well below target
    for _ in range(10):
        plan.record(0.1)
    assert plan.next_stake(0) == 3.0


def test_anti_martingale_presses_wins_and_resets_on_loss():
    plan = AntiMartingale(base=2.0, factor=2.0, max_steps=2)
    assert plan.next_stake(0) == 2.0
    plan.record(1)
    assert plan.next_stake(0) == 4.0
    plan.record(1)
    plan.record(1)
    assert plan.next_stake(0) == 8.0   # capped at max_steps
    plan.record(-1)
    assert plan.next_stake(0) == 2.0


def test_make_plan():
    assert isinstance(make_plan("fixed_fraction", fraction=0.02), FixedFraction)
    with pytest.raises(ValueError, match="unknown staking plan"):
        make_plan("martingale")