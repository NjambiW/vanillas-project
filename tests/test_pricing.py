import math

import pytest

from pricing import (
    bs_call,
    bs_put,
    expected_move,
    greeks,
    implied_vol,
    index_vol,
    years,
)


def test_atm_call_reference_value():
    # S=K=100, T=1y, vol 20%, r=0  ->  about 7.9656
    assert bs_call(100, 100, 1.0, 0.2) == pytest.approx(7.9656, abs=1e-3)


def test_put_call_parity_with_zero_rate():
    S, K, T, v = 105.0, 100.0, 0.5, 0.3
    assert bs_call(S, K, T, v) - bs_put(S, K, T, v) == pytest.approx(S - K, abs=1e-9)


def test_expired_option_is_intrinsic_value():
    assert bs_call(110, 100, 0, 0.2) == 10
    assert bs_put(110, 100, 0, 0.2) == 0


def test_price_rises_with_volatility_and_time():
    assert bs_call(100, 100, 1, 0.3) > bs_call(100, 100, 1, 0.2)
    assert bs_call(100, 100, 2, 0.2) > bs_call(100, 100, 1, 0.2)


def test_deep_otm_is_nearly_worthless():
    assert bs_call(100, 200, 0.01, 0.2) < 1e-9


def test_greeks_atm():
    g = greeks("call", 100, 100, 1.0, 0.2)
    assert g["delta"] == pytest.approx(0.5398, abs=1e-3)
    assert g["gamma"] > 0 and g["vega"] > 0 and g["theta"] < 0
    assert greeks("put", 100, 100, 1.0, 0.2)["delta"] == pytest.approx(g["delta"] - 1)


def test_implied_vol_roundtrip():
    price = bs_put(1118.78, 1114.38, 5 / 525600, 1.0)
    iv = implied_vol(price, "put", 1118.78, 1114.38, 5 / 525600)
    assert iv == pytest.approx(1.0, abs=1e-6)


def test_implied_vol_nan_when_price_impossible():
    assert math.isnan(implied_vol(1000.0, "call", 100, 100, 1.0))


def test_years_conversion():
    assert years(1, "d") == pytest.approx(1 / 365)
    assert years(5, "m") == pytest.approx(5 * 60 / (365 * 86400))
    with pytest.raises(ValueError):
        years(5, "t")


def test_index_vol_from_symbol():
    assert index_vol("1HZ100V") == 1.0
    assert index_vol("R_75") == 0.75
    assert index_vol("1HZ10V") == 0.10
    with pytest.raises(ValueError):
        index_vol("EURUSD")


def test_expected_move():
    assert expected_move(100, 0.2, 1.0) == pytest.approx(20.0)