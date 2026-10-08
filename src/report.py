"""Write reports/backtest_report.json, the file demo_review.py and go_live_gate.py read.

Call it at the end of your own run_backtest.py for the ONE strategy and hold you want to take
forward. Segment dicts hold plain numbers:

    trades, win_rate, roi (return per 1.0 staked AFTER markup), gross_roi, avg_markup_pct,
    profit_factor, max_drawdown (in stake units), p_value, and optionally ci_low / ci_high
    (the block-bootstrap interval of the mean net return).

`verdict` is your pass/fail label ("fail", "postpone", "candidate", "confirmed"); the go-live
gate only accepts "confirmed". `variants_tested` is how many strategy/setting combinations you
have tried IN TOTAL; the gate divides its 0.05 significance bar by it.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def _clean(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def build_report(*, strategy, hold_candles, symbol, candle_seconds, cost_model_source,
                 premium_gate_pct, risk_rules_applied, variants_tested, train, out_of_sample,
                 verdict=None, extra=None) -> dict:
    report = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy": strategy,
        "hold_candles": hold_candles,
        "symbol": symbol,
        "candle_seconds": candle_seconds,
        "cost_model_source": cost_model_source,
        "premium_gate_pct": premium_gate_pct,
        "risk_rules_applied": risk_rules_applied,
        "variants_tested": max(1, int(variants_tested)),
        "train": dict(train),
        "out_of_sample": dict(out_of_sample),
    }
    if verdict:
        report["verdict"] = verdict
    if extra:
        report.update(extra)
    return _clean(report)


def write_report(report: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(report), indent=2), encoding="utf-8")
    return path