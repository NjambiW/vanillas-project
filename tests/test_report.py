import json
import math
import tempfile
from pathlib import Path

from backtest.report import build_report, write_report


def test_report_round_trips_and_removes_values_json_cannot_hold():
    report = build_report(
        strategy="trend_pullback", hold_candles=5, symbol="1HZ100V", candle_seconds=60,
        cost_model_source="logs/markup_x.csv", premium_gate_pct=15.0, risk_rules_applied=True,
        variants_tested=8, train={"roi": -0.04, "profit_factor": math.inf},
        out_of_sample={"roi": -0.125, "p_value": 0.63, "ci_low": -0.24}, verdict="fail")
    with tempfile.TemporaryDirectory() as tmp:
        path = write_report(report, Path(tmp) / "reports" / "backtest_report.json")
        loaded = json.loads(path.read_text())
    assert loaded["train"]["profit_factor"] is None          # inf became null
    assert loaded["variants_tested"] == 8 and loaded["verdict"] == "fail"
    assert loaded["out_of_sample"]["ci_low"] == -0.24