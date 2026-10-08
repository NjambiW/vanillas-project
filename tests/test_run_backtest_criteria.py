import argparse

import pandas as pd

from backtest.cost_model import LEVELS, MEASURED, CostModel
from run_backtest import (StressedCostModel, fold_checks, judge, make_segments, mark,
                          window_problem)

MIN = 300


def args(**overrides):
    base = dict(walk_forward=False, split=0.7, folds=4)
    base.update(overrides)
    return argparse.Namespace(**base)


def verdict(**overrides):
    base = dict(independent=500, ci_ok=True, stable=True, dominant=False,
                checkable=True, hold_ok=True, hold_applies=True,
                stress_ok=True, stress_applies=True, quotes_ok=True,
                min_oos_trades=MIN)
    base.update(overrides)
    return judge(**base)[0]


def test_a_thin_sample_is_postponed_rather_than_failed():
    assert verdict(independent=MIN - 1) == "postpone"


def test_every_criterion_must_pass_before_promotion():
    assert verdict() == "CONFIRMED"


def test_no_replay_is_a_candidate_not_a_confirmation():
    assert verdict(quotes_ok=None) == "CANDIDATE*"


def test_a_failing_quote_replay_fails_the_promotion():
    assert verdict(quotes_ok=False) == "fail"


def test_a_negative_holdout_fails_the_promotion():
    assert verdict(hold_ok=False, hold_applies=True) == "fail"


def test_an_unscored_holdout_downgrades_to_a_candidate():
    assert verdict(hold_ok=False, hold_applies=False) == "CANDIDATE*"


def test_a_negative_stress_rerun_fails_the_promotion():
    assert verdict(stress_ok=False, stress_applies=True) == "fail"


def test_an_unapplied_stress_downgrades_to_a_candidate():
    assert verdict(stress_ok=False, stress_applies=False) == "CANDIDATE*"


def test_every_check_must_have_run_before_the_word_confirmed():
    assert verdict(hold_applies=False) == "CANDIDATE*"
    assert verdict(stress_applies=False) == "CANDIDATE*"
    assert verdict(quotes_ok=None) == "CANDIDATE*"
    assert verdict() == "CONFIRMED"


def test_a_negative_confidence_interval_fails():
    assert verdict(ci_ok=False) == "fail"


def test_profit_that_is_unstable_or_dominated_fails():
    assert verdict(stable=False) == "fail"
    assert verdict(dominant=True, checkable=True) == "fail"


def test_dominance_cannot_be_judged_with_a_single_window():
    assert verdict(dominant=True, checkable=False) == "CONFIRMED"


def test_flags_say_which_criteria_applied():
    _, flags = judge(500, True, True, False, True, False, False, False, False,
                     None, MIN)
    assert flags["enough"] is True
    assert flags["hold_applies"] is False
    assert flags["quotes"] is None


def test_mark_only_renders_not_applicable_when_it_does_not_apply():
    assert mark(True) == "ok"
    assert mark(False) == "NO"
    assert mark(False, applies=False) == "n/a"
    assert mark(True, applies=False) == "n/a"


def test_stressed_cost_model_adds_points_to_every_quote():
    base = CostModel({60: MEASURED[60]})
    stressed = StressedCostModel(base, 3.0)
    for level in LEVELS:
        assert stressed.markup_pct(60, level) == base.markup_pct(60, level) + 3.0


def test_stress_of_zero_leaves_the_quotes_untouched():
    base = CostModel()
    assert StressedCostModel(base, 0.0).markup_pct(300, 0.0) == base.markup_pct(300, 0.0)


def test_fold_checks_on_a_single_split():
    def result(roi, p):
        return ({"roi": roi, "total_profit": roi}, {"p_value": p})

    both = {"train": result(0.10, 0.01), "test": result(0.20, 0.01)}
    assert fold_checks(both, ["test"], walk_forward=False) == (True, False, False)

    one_bad = {"train": result(0.10, 0.01), "test": result(-0.20, 0.01)}
    assert fold_checks(one_bad, ["test"], walk_forward=False)[0] is False

    noisy = {"train": result(0.10, 0.40), "test": result(0.20, 0.01)}
    assert fold_checks(noisy, ["test"], walk_forward=False)[0] is False


def test_fold_checks_wants_a_majority_of_windows_to_pay():
    def fold(profit):
        return ({"total_profit": profit}, {"p_value": 0.5})

    spread = {"fold1": fold(4.0), "fold2": fold(4.0), "fold3": fold(4.0)}
    assert fold_checks(spread, ["fold1", "fold2", "fold3"], True)[:2] == (True, False)

    mostly_losing = {"fold1": fold(10.0), "fold2": fold(-1.0), "fold3": fold(-1.0)}
    assert fold_checks(mostly_losing, ["fold1", "fold2", "fold3"], True)[0] is False

    losing = {"fold1": fold(-10.0), "fold2": fold(-1.0), "fold3": fold(-1.0)}
    assert fold_checks(losing, ["fold1", "fold2", "fold3"], True)[0] is False


def test_one_window_carrying_all_the_profit_is_flagged_as_dominance():
    def fold(profit):
        return ({"total_profit": profit}, {"p_value": 0.5})

    lucky = {"fold1": fold(100.0), "fold2": fold(1.0), "fold3": fold(1.0)}
    stable, dominant, checkable = fold_checks(
        lucky, ["fold1", "fold2", "fold3"], True)
    assert stable and dominant and checkable

    single = {"fold1": fold(100.0)}
    assert fold_checks(single, ["fold1"], True)[1] is False


# ------------------------------------------------------------------ segmentation
def candles(n):
    return pd.DataFrame({"close": range(n)})


def test_the_single_split_purges_the_max_holding_period_before_testing():
    n, hold = 20_000, 240
    segments, oos = make_segments(candles(n), args(), hold)
    assert oos == ["test"]
    train_end = len(segments["train"])
    test_start = n - len(segments["test"])
    assert test_start - train_end == hold          # no training label reaches the test window


def test_walk_forward_yields_exactly_the_requested_number_of_folds():
    n = 20_000
    for hold in (1, 60, 240, 1440):
        segments, oos = make_segments(candles(n), args(walk_forward=True, folds=4), hold)
        assert len(oos) == 4, f"hold={hold} gave {len(oos)} folds"


def test_walk_forward_keeps_a_purge_between_training_and_the_first_fold():
    n, hold = 20_000, 1440
    segments, _ = make_segments(candles(n), args(walk_forward=True, folds=4), hold)
    train_end = int(n * 0.7)
    first = segments[list(segments)[0]]
    offset = int(first["close"].iloc[0])           # absolute index of the fold's first row
    assert offset - train_end == hold


def test_a_hold_that_fits_the_windows_reports_no_problem():
    assert window_problem(candles(20_000), args(walk_forward=True, folds=4), 60) is None
    assert window_problem(candles(20_000), args(), 240) is None


def test_a_hold_longer_than_its_test_window_is_called_out_rather_than_run():
    # 4 folds of 1140 candles: a 1440-candle contract can never expire inside one,
    # and the run used to print "(no trades)" as though the strategy had stayed quiet.
    why = window_problem(candles(20_000), args(walk_forward=True, folds=4), 1440)
    assert why is not None and "expire" in why


def test_a_hold_that_would_eat_the_whole_split_is_called_out():
    why = window_problem(candles(20_000), args(split=0.7), 6_000)
    assert why is not None


def test_fold_geometry_follows_the_hold_not_the_run():
    # The shortest and longest holds get different windows, so the purge is never
    # sized by a hold this strategy does not use.
    small, _ = make_segments(candles(20_000), args(walk_forward=True, folds=4), 60)
    large, _ = make_segments(candles(20_000), args(walk_forward=True, folds=4), 1440)
    assert len(small["fold1"]) > len(large["fold1"])
