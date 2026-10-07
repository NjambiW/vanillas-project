"""
Phase 9: go-live gate. All four checks must pass before any real money.

Run from the project root, after:
    python src/run_backtest.py --strategy NAME --hold N --export     (out-of-sample report)
    python src/demo_review.py                                          (100+ demo trades)
    python src/kill_switch_drill.py
then:
    python src/go_live_gate.py

Checks:
  1. A real edge after markup, out of sample: reports/backtest_report.json must show, on data the
     strategy never saw, 100+ trades, positive return per stake, and a p-value below 0.05 divided
     by the number of variants you have tried (so trying many things does not make luck look
     like skill); the training half must also be positive; and the cost model must come from a
     measure_markup.py CSV, not the built-in table.
  2. Demo results consistent with backtest: 100+ settled demo trades, same strategy and expiry,
     and no HIGH flag in demo_review.py.
  3. Kill switch tested on purpose: the drill passed, is under 30 days old, and was run with the
     SAME risk settings as config.py now has.
  4. Real stake at the minimum: read from config.py (STAKING_PLAN = "fixed", FIXED_STAKE at or
     below MIN_STAKE). It is not typed in by hand, so it cannot be fudged.

Writes reports/go_live_gate.json. Exit code 0 only if every gate passes. This gate is advisory:
it cannot stop you pointing the bot at a real account, so it only counts if you respect it.
"""
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from demo_review import MIN_TRADES, review  # noqa: E402

MIN_OOS_TRADES = 100
MAX_AGE_DAYS = 30
BASE_ALPHA = 0.05


def _now():
    return datetime.now(timezone.utc)


def _parse_stamp(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _resolve(path, root):
    p = Path(path)
    return p if p.is_absolute() else root / p


def gate_oos(bt_path, root=ROOT):
    p = _resolve(bt_path, root)
    if not p.exists():
        return False, f"Missing {bt_path}. Create it: python src/run_backtest.py --strategy NAME --hold N --export"
    d = json.loads(p.read_text(encoding="utf-8"))
    oos, train = d.get("out_of_sample"), d.get("train")
    if not isinstance(oos, dict) or not isinstance(train, dict):
        return False, "Report has no train / out_of_sample sections. Re-create it with run_backtest.py --export."
    if str(d.get("cost_model_source", "")).startswith("built-in"):
        return False, "The backtest used the built-in markup table. Run measure_markup.py first, then re-run the backtest."
    try:
        age = _now() - _parse_stamp(d["generated"])
        if age > timedelta(days=MAX_AGE_DAYS):
            return False, f"Backtest report is {age.days} days old; re-run it on fresh data."
    except (KeyError, ValueError):
        return False, "Backtest report has no valid timestamp."

    n, roi, p_value = oos.get("trades"), oos.get("roi"), oos.get("p_value")
    if n is None or roi is None or p_value is None:
        return False, "Out-of-sample section is missing trades, roi or p_value."
    if n < MIN_OOS_TRADES:
        return False, f"Only {int(n)} out-of-sample trades (need {MIN_OOS_TRADES}+)."
    variants = max(1, int(d.get("variants_tested") or 1))
    alpha = BASE_ALPHA / variants
    if roi <= 0:
        return False, f"Out-of-sample return per stake is {roi:+.1%} after markup. Not profitable."
    if p_value >= alpha:
        return False, (f"Out-of-sample return {roi:+.1%} is positive but p = {p_value:.3f} is not below "
                       f"{alpha:.4f} (0.05 / {variants} variants tried). Could easily be luck.")
    if (train.get("roi") or 0) <= 0:
        return False, f"Out-of-sample looks good but the training half returned {train.get('roi', 0):+.1%}."
    return True, f"OOS return {roi:+.1%} over {int(n)} trades, p = {p_value:.4f} < {alpha:.4f}, training also positive."


def gate_consistency(journal, bt_path, root=ROOT):
    try:
        res = review(journal, str(_resolve(bt_path, root)), {})
    except SystemExit as e:
        return False, f"Demo review failed: {e}"
    n = res["overall"]["trades"]
    if n < MIN_TRADES:
        return False, f"{n}/{MIN_TRADES} demo trades logged."
    if not res["backtest"]:
        return False, "No backtest numbers to compare against."
    highs = [f["message"] for f in res["flags"] if f["severity"] == "HIGH"]
    if highs:
        return False, "Demo and backtest do not agree: " + " | ".join(highs)
    return True, f"{n} demo trades, no high-severity gap vs backtest."


def gate_kill_switch(root=ROOT, fingerprint=None):
    p = root / "reports" / "kill_switch_drill.json"
    if not p.exists():
        return False, "No drill result. Run python src/kill_switch_drill.py."
    d = json.loads(p.read_text(encoding="utf-8"))
    if not d.get("passed"):
        return False, f"Last drill failed: {d.get('error', 'unknown')}"
    try:
        age = _now() - _parse_stamp(d["ran_at"])
    except (KeyError, ValueError):
        return False, "Drill result has no valid timestamp."
    if age > timedelta(days=MAX_AGE_DAYS):
        return False, f"Drill is {age.days} days old; re-run it."
    if fingerprint is None:
        try:
            from risk import settings_fingerprint
            fingerprint = settings_fingerprint()
        except Exception as exc:  # noqa: BLE001
            return False, f"Could not read config.py ({exc})."
    if d.get("config_fingerprint") != fingerprint:
        return False, "Your risk settings changed since the drill ran. Re-run kill_switch_drill.py."
    return True, f"Drill passed under today's settings (halted after {d.get('halted_after_losses')} losses)."


def gate_stake(settings=None):
    if settings is None:
        try:
            import config
            settings = {"plan": config.STAKING_PLAN, "fixed_stake": config.FIXED_STAKE,
                        "min_stake": config.MIN_STAKE, "dry_run": config.DRY_RUN}
        except Exception as exc:  # noqa: BLE001
            return False, f"Could not read config.py ({exc})."
    if settings["plan"] != "fixed":
        return False, f"STAKING_PLAN is '{settings['plan']}'. Use 'fixed' for the first real money."
    if settings["fixed_stake"] > settings["min_stake"] + 1e-9:
        return False, f"FIXED_STAKE {settings['fixed_stake']} is above the minimum {settings['min_stake']}."
    return True, f"Fixed stake {settings['fixed_stake']} (minimum {settings['min_stake']})."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal")
    ap.add_argument("--backtest", default="reports/backtest_report.json")
    a = ap.parse_args()

    checks = [
        ("Real edge after markup, out of sample", gate_oos(a.backtest)),
        ("Demo results consistent with backtest", gate_consistency(a.journal, a.backtest)),
        ("Kill switch tested on purpose", gate_kill_switch()),
        ("Real stake at the minimum", gate_stake()),
    ]
    out = []
    for name, (ok, why) in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {why}")
        out.append({"gate": name, "passed": ok, "detail": why})
    all_ok = all(c["passed"] for c in out)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "go_live_gate.json").write_text(json.dumps(
        {"generated": _now().strftime("%Y-%m-%dT%H:%M:%SZ"), "all_passed": all_ok, "checks": out}, indent=2),
        encoding="utf-8")
    print("\nGO-LIVE GATE:", "OPEN" if all_ok else "CLOSED (no real money yet)")
    print("Reminder: scale the stake only after weeks of live data.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()