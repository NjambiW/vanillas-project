"""
Phase 9: go-live gate. All four checks must pass before any real money.

Run from the project root (after demo_review.py and kill_switch_drill.py):
    python src/go_live_gate.py --real-stake 0.50 --min-stake 0.50

Checks:
  1. Positive expectancy after markup, out of sample   (reports/backtest_report.json)
  2. Demo results consistent with backtest             (demo_review.py, 100+ trades, no HIGH flags)
  3. Kill switch tested on purpose                     (reports/kill_switch_drill.json, passed, <= 30 days old)
  4. Real stake set at the minimum                     (--real-stake <= --min-stake)

Writes reports/go_live_gate.json. Exit code 0 only if every gate passes.
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from demo_review import MIN_TRADES, load_backtest, review  # noqa: E402

MIN_OOS_TRADES = 100
DRILL_MAX_AGE_DAYS = 30


def gate_oos(bt_path):
    p = ROOT / bt_path
    if not p.exists():
        return False, f"Missing {bt_path}. Export the out-of-sample report from run_backtest.py."
    bt = load_backtest(bt_path, {})
    exp, n = bt.get("expectancy"), bt.get("trades")
    if exp is None:
        return False, "No expectancy in the backtest report."
    if n is not None and n < MIN_OOS_TRADES:
        return False, f"Only {int(n)} out-of-sample trades (need {MIN_OOS_TRADES}+); expectancy {exp:+.4f}."
    ok = exp > 0
    return ok, f"OOS expectancy after markup {exp:+.4f}" + (f" over {int(n)} trades" if n else "")


def gate_consistency(journal, bt_path):
    try:
        res = review(journal, bt_path, {})
    except SystemExit as e:
        return False, f"Demo review failed: {e}"
    n = res["overall"]["trades"]
    if n < MIN_TRADES:
        return False, f"{n}/{MIN_TRADES} demo trades logged."
    if not res["backtest"]:
        return False, "No backtest numbers to compare against."
    highs = [f["message"] for f in res["flags"] if f["severity"] == "HIGH"]
    if highs:
        return False, "Large demo/backtest gap: " + " | ".join(highs)
    return True, f"{n} demo trades, no high-severity gap vs backtest."


def gate_kill_switch():
    p = ROOT / "reports" / "kill_switch_drill.json"
    if not p.exists():
        return False, "No drill result. Run python src/kill_switch_drill.py."
    d = json.loads(p.read_text(encoding="utf-8"))
    if not d.get("passed"):
        return False, f"Last drill failed: {d.get('error', 'unknown')}"
    try:
        ran = datetime.strptime(d["ran_at"].rstrip("Z"), "%Y-%m-%dT%H:%M:%S")
    except (KeyError, ValueError):
        return False, "Drill result has no valid timestamp."
    if datetime.utcnow() - ran > timedelta(days=DRILL_MAX_AGE_DAYS):
        return False, f"Drill is older than {DRILL_MAX_AGE_DAYS} days; re-run it."
    return True, f"Drill passed (halted after {d.get('halted_after_losses')} losses, stayed halted)."


def gate_stake(real_stake, min_stake):
    if real_stake is None or min_stake is None:
        return False, "Pass --real-stake and --min-stake."
    ok = real_stake <= min_stake + 1e-9
    return ok, f"Real stake {real_stake} vs minimum {min_stake}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal")
    ap.add_argument("--backtest", default="reports/backtest_report.json")
    ap.add_argument("--real-stake", type=float)
    ap.add_argument("--min-stake", type=float)
    a = ap.parse_args()

    checks = [
        ("Positive expectancy after markup, out of sample", gate_oos(a.backtest)),
        ("Demo results consistent with backtest", gate_consistency(a.journal, a.backtest)),
        ("Kill switch tested on purpose", gate_kill_switch()),
        ("Real stake at the minimum", gate_stake(a.real_stake, a.min_stake)),
    ]
    out = []
    for name, (ok, why) in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {why}")
        out.append({"gate": name, "passed": ok, "detail": why})
    all_ok = all(c["passed"] for c in out)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "go_live_gate.json").write_text(json.dumps(
        {"generated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
         "all_passed": all_ok, "checks": out}, indent=2), encoding="utf-8")
    print("\nGO-LIVE GATE:", "OPEN" if all_ok else "CLOSED (no real money yet)")
    print("Reminder: scale the stake only after weeks of live data.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()