import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from go_live_gate import gate_kill_switch, gate_oos, gate_stake


def stamp(days_ago=0):
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def good_report(**over):
    report = {
        "generated": stamp(), "strategy": "trend_pullback", "hold_candles": 5,
        "cost_model_source": "logs/markup_20261006.csv (+ built-in table for missing expiries)",
        "variants_tested": 1,
        "train": {"roi": 0.05, "p_value": 0.03},
        "out_of_sample": {"trades": 150, "roi": 0.06, "p_value": 0.02},
    }
    report.update(over)
    return report


def oos(tmp, report):
    (Path(tmp) / "reports").mkdir()
    (Path(tmp) / "reports" / "backtest_report.json").write_text(json.dumps(report))
    return gate_oos("reports/backtest_report.json", root=Path(tmp))


def test_a_significant_positive_result_on_unseen_data_passes():
    with tempfile.TemporaryDirectory() as tmp:
        ok, why = oos(tmp, good_report())
    assert ok, why


def test_positive_but_not_significant_fails():
    with tempfile.TemporaryDirectory() as tmp:
        ok, why = oos(tmp, good_report(out_of_sample={"trades": 150, "roi": 0.06, "p_value": 0.30}))
    assert not ok and "luck" in why


def test_trying_many_variants_makes_the_bar_higher():
    with tempfile.TemporaryDirectory() as tmp:
        ok, why = oos(tmp, good_report(variants_tested=10))      # p = 0.02 is not below 0.05 / 10
    assert not ok and "variants" in why


def test_negative_return_fails():
    with tempfile.TemporaryDirectory() as tmp:
        ok, why = oos(tmp, good_report(out_of_sample={"trades": 150, "roi": -0.15, "p_value": 0.01}))
    assert not ok and "Not profitable" in why


def test_too_few_trades_fails():
    with tempfile.TemporaryDirectory() as tmp:
        ok, why = oos(tmp, good_report(out_of_sample={"trades": 40, "roi": 0.2, "p_value": 0.001}))
    assert not ok and "40" in why


def test_built_in_cost_table_fails():
    with tempfile.TemporaryDirectory() as tmp:
        ok, why = oos(tmp, good_report(cost_model_source="built-in table (measured 2026-10-06)"))
    assert not ok and "measure_markup" in why


def test_losing_training_half_fails():
    with tempfile.TemporaryDirectory() as tmp:
        ok, why = oos(tmp, good_report(train={"roi": -0.1, "p_value": 0.5}))
    assert not ok and "training" in why


def test_stale_report_fails():
    with tempfile.TemporaryDirectory() as tmp:
        ok, why = oos(tmp, good_report(generated=stamp(days_ago=45)))
    assert not ok and "old" in why


def test_missing_report_fails_with_instructions():
    with tempfile.TemporaryDirectory() as tmp:
        ok, why = gate_oos("reports/backtest_report.json", root=Path(tmp))
    assert not ok and "--export" in why


def drill(tmp, **over):
    data = {"passed": True, "ran_at": stamp(), "config_fingerprint": "abc123", "halted_after_losses": 6}
    data.update(over)
    (Path(tmp) / "reports").mkdir(exist_ok=True)
    (Path(tmp) / "reports" / "kill_switch_drill.json").write_text(json.dumps(data))


def test_kill_switch_gate_needs_a_fresh_drill_under_the_same_settings():
    with tempfile.TemporaryDirectory() as tmp:
        drill(tmp)
        assert gate_kill_switch(Path(tmp), fingerprint="abc123")[0]
        ok, why = gate_kill_switch(Path(tmp), fingerprint="different")
        assert not ok and "settings changed" in why
        drill(tmp, ran_at=stamp(days_ago=40))
        assert not gate_kill_switch(Path(tmp), fingerprint="abc123")[0]
        drill(tmp, passed=False, error="STOP kill file failed")
        assert not gate_kill_switch(Path(tmp), fingerprint="abc123")[0]


def test_stake_gate_reads_config_values():
    assert gate_stake({"plan": "fixed", "fixed_stake": 1.0, "min_stake": 1.0})[0]
    assert not gate_stake({"plan": "fixed_fraction", "fixed_stake": 1.0, "min_stake": 1.0})[0]
    assert not gate_stake({"plan": "fixed", "fixed_stake": 5.0, "min_stake": 1.0})[0]