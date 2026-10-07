import json
import tempfile
from pathlib import Path

from demo_review import compute_normalised, duration_text, review
from journal import Journal


def make_journal(tmp, rois, stake=25.0, markup=11.0, strategy="trend_pullback", duration="5m"):
    path = Path(tmp) / "journal.db"
    j = Journal(path)
    for i, r in enumerate(rois):
        t = j.open_trade(strategy=strategy, contract_id=i, contract_type="VANILLALONGCALL",
                         duration=duration, stake=stake, markup_pct=markup)
        j.close_trade(t, r * stake, "won" if r > 0 else "lost")
    j.close()
    return path


def make_report(tmp, **over):
    report = {"strategy": "trend_pullback", "hold_candles": 5, "candle_seconds": 60,
              "out_of_sample": {"roi": -0.30, "win_rate": 0.17, "avg_markup_pct": 11.0,
                                "max_drawdown": 60.0, "trades": 150}}
    report.update(over)
    path = Path(tmp) / "backtest_report.json"
    path.write_text(json.dumps(report))
    return str(path)


def flags(res, severity):
    return [f["message"] for f in res["flags"] if f["severity"] == severity]


def test_normalised_metrics_are_per_stake_not_per_dollar():
    out = compute_normalised([20.0, -10.0, 5.0], [10.0, 10.0, 5.0], [10.0, 10.0, 10.0])
    assert abs(out["roi"] - (2.0 - 1.0 + 1.0) / 3) < 1e-9
    assert abs(out["max_drawdown_units"] - 1.0) < 1e-9
    assert out["avg_markup_pct"] == 10.0


def test_big_demo_stakes_do_not_create_a_false_gap_with_a_stake_1_backtest():
    rois = [-1.0] * 100 + [3.0] * 20            # 120 trades, return -33% per stake
    with tempfile.TemporaryDirectory() as tmp:
        res = review(str(make_journal(tmp, rois, stake=25.0)), make_report(tmp), {})
    assert abs(res["overall"]["roi"] - (-40 / 120)) < 1e-9
    assert flags(res, "HIGH") == []


def test_different_strategy_in_demo_is_a_high_flag():
    with tempfile.TemporaryDirectory() as tmp:
        res = review(str(make_journal(tmp, [-1.0] * 30, strategy="breakout")), make_report(tmp), {})
    assert any("cannot be compared" in m for m in flags(res, "HIGH"))


def test_return_changing_sign_is_a_high_flag():
    report_roi = {"roi": 0.10, "win_rate": 0.5, "max_drawdown": 60.0, "avg_markup_pct": 11.0, "trades": 150}
    with tempfile.TemporaryDirectory() as tmp:
        report = make_report(tmp, out_of_sample=report_roi)
        res = review(str(make_journal(tmp, [-1.0] * 100 + [2.0] * 20)), report, {})
    assert any("changes sign" in m for m in flags(res, "HIGH"))


def test_cost_model_drift_is_flagged():
    with tempfile.TemporaryDirectory() as tmp:
        res = review(str(make_journal(tmp, [-1.0] * 100 + [3.0] * 20, markup=25.0)), make_report(tmp), {})
    assert any("cost model has drifted" in m for m in flags(res, "MED"))


def test_wrong_expiry_is_flagged():
    with tempfile.TemporaryDirectory() as tmp:
        res = review(str(make_journal(tmp, [-1.0] * 100 + [3.0] * 20, duration="15m")), make_report(tmp), {})
    assert any("Backtest holds 5m" in m for m in flags(res, "MED"))


def test_no_backtest_report_is_explained_not_a_crash():
    with tempfile.TemporaryDirectory() as tmp:
        res = review(str(make_journal(tmp, [-1.0] * 30)), str(Path(tmp) / "missing.json"), {})
    assert any("run_backtest.py" in m for m in flags(res, "INFO"))


def test_duration_text():
    assert duration_text(300) == "5m" and duration_text(3600) == "1h" and duration_text(45) == "45s"