import math

import pytest

from markup import MarkupError, find_field, markup_row
from pricing import bs_call, years


def test_find_field_searches_nested_data():
    data = {"a": 1, "inner": {"deep": [{"payout_per_point": "0.5"}]}}
    assert find_field(data, ("payout_per_point",)) == "0.5"
    assert find_field(data, ("missing",)) is None


def test_zero_markup_when_price_equals_model():
    spot, barrier, T = 1000.0, "+2.30", years(5, "m")
    k = 0.8
    fair = k * bs_call(spot, spot + 2.30, T, 1.0)
    proposal = {"ask_price": fair, "payout_per_point": k, "spot": spot}
    row = markup_row("VANILLALONGCALL", proposal, spot, barrier, 5, "m", "1HZ100V")
    assert row["markup_pct"] == pytest.approx(0.0, abs=0.01)
    assert row["implied_vol"] == pytest.approx(1.0, abs=1e-3)


def test_markup_is_measured_against_fair_price():
    spot, T, k = 1000.0, years(5, "m"), 0.8
    fair = k * bs_call(spot, spot, T, 1.0)
    proposal = {"ask_price": fair * 1.10, "payout_per_point": k, "spot": spot}
    row = markup_row("VANILLALONGCALL", proposal, spot, "+0.00", 5, "m", "1HZ100V")
    assert row["markup_pct"] == pytest.approx(10.0, abs=0.01)
    assert row["implied_vol"] > 1.0  # paying extra looks like extra volatility


def test_fields_found_inside_nested_contract_details():
    spot, T, k = 1000.0, years(5, "m"), 1.0
    proposal = {"ask_price": 1.0, "contract_details": {"payout_per_point": k}}
    row = markup_row("VANILLALONGPUT", proposal, spot, "-2.30", 5, "m", "1HZ100V")
    assert row["payout_per_point"] == 1.0 and row["type"] == "VANILLALONGPUT"


def test_missing_payout_per_point_raises_helpfully():
    with pytest.raises(MarkupError, match="payout_per_point"):
        markup_row("VANILLALONGCALL", {"ask_price": 1.0}, 1000.0, "+0.00", 5, "m", "1HZ100V")


# A real proposal from Deriv (1HZ100V, 1 minute Call, barrier +2.20, stake 1.0)
REAL_PROPOSAL = {
    "ask_price": 1,
    "barrier_choices": ["+2.20", "+1.10", "+0.00", "-1.10", "-2.20"],
    "contract_details": {"barrier": "1226.39"},
    "display_number_of_contracts": "9.155054",
    "min_stake": 0.4,
    "max_stake": 567,
    "payout": 0,
    "spot": 1224.19,
}


def test_real_deriv_proposal_shape_is_understood():
    row = markup_row("VANILLALONGCALL", REAL_PROPOSAL, 1224.04, "+2.20", 1, "m", "1HZ100V")
    assert row["payout_per_point"] == 9.155054
    assert row["strike"] == 1226.39            # taken from contract_details, not recomputed
    assert row["spot"] == 1224.19
    assert 42.0 < row["markup_pct"] < 43.0     # worked out independently: about 42.5%
    assert 1.10 < row["implied_vol"] < 1.11    # Deriv's price implies about 110% vol


def test_absolute_barrier_without_contract_details_is_used_as_the_strike():
    spot, k = 1000.0, 1.0
    proposal = {"ask_price": 1.0, "display_number_of_contracts": k, "spot": spot}
    row = markup_row("VANILLALONGCALL", proposal, spot, "1020.00", 1, "d", "1HZ100V")
    assert row["strike"] == 1020.0