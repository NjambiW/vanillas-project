"""
Phase 9: kill switch tested on purpose.

Drives the REAL RiskManager (src/risk.py) with your config.py limits and a fake clock, and
checks every stop mechanism:
  1. consecutive-loss halt fires at exactly max_consecutive_losses and stays on
  2. daily-loss-limit halt fires and stays on
  3. manual halt() (used by the executor on errors) blocks trading
  4. the STOP kill file blocks trading, and trading resumes when the file is deleted
  5. an automatic halt clears at the next UTC day
  6. only max_open_positions trades can be open at once

It also records WARNINGS (not failures), e.g. whether a halt survives a bot restart.

Run from the project root:
    python src/kill_switch_drill.py

Writes reports/kill_switch_drill.json, which go_live_gate.py reads.
"""
import json
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from risk import RiskConfig, RiskManager  # noqa: E402
from staking import make_plan  # noqa: E402

BALANCE = 10_000.0


class Clock:
    def __init__(self):
        self.t = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t

    def next_day(self):
        self.t = (self.t + timedelta(days=1)).replace(hour=1, minute=0)


def make_manager(clock, **overrides):
    try:
        import config
        plan_name = config.STAKING_PLAN
        params = {"fraction": config.RISK_FRACTION} if plan_name == "fixed_fraction" else {}
    except Exception:  # noqa: BLE001
        plan_name, params = "fixed_fraction", {"fraction": 0.01}
    base = RiskConfig.from_settings()
    cfg = replace(base, min_seconds_between_trades=0, **overrides)
    return RiskManager(cfg, make_plan(plan_name, **params), now=clock), cfg


def play_losses_until_blocked(rm, balance, limit=500):
    """Open/close losing trades until approve_trade refuses. Returns (trades, decision, balance)."""
    count = 0
    while count < limit:
        d = rm.approve_trade(balance)
        if not d.allowed:
            return count, d, balance
        rm.on_trade_opened(d.stake)
        rm.on_trade_closed(-d.stake)
        balance -= d.stake
        count += 1
    return count, None, balance


# ------------------------------------------------------------------ checks
def check_consecutive(extra):
    clock = Clock()
    rm, cfg = make_manager(clock, daily_loss_limit_pct=100.0, max_trades_per_day=100000)
    count, d, bal = play_losses_until_blocked(rm, BALANCE)
    if d is None:
        return False, f"never halted after {count} losses"
    if count != cfg.max_consecutive_losses:
        return False, f"halted after {count} losses, expected {cfg.max_consecutive_losses}"
    if "losses in a row" not in d.reason:
        return False, f"halted for the wrong reason: {d.reason}"
    if any(rm.approve_trade(bal).allowed for _ in range(3)):
        return False, "halt did not persist across repeated approve_trade calls"
    extra["halted_after_losses"] = count
    # restart behaviour, informational
    rm2, _ = make_manager(clock, daily_loss_limit_pct=100.0, max_trades_per_day=100000)
    if rm2.approve_trade(bal).allowed:
        extra.setdefault("warnings", []).append(
            "A halt is held in memory only: restarting the bot clears the halt, the daily loss "
            "tally and the consecutive-loss count."
        )
    return True, f"halted after {count} consecutive losses and stayed halted"


def check_daily_loss(extra):
    clock = Clock()
    rm, cfg = make_manager(clock, max_consecutive_losses=100000, max_trades_per_day=100000)
    balance, trades, d = BALANCE, 0, None
    while trades < 5000:
        # two losses then a small win, so the consecutive limit never matters
        for profit_factor in (-1.0, -1.0, 0.3):
            d = rm.approve_trade(balance)
            if not d.allowed:
                break
            rm.on_trade_opened(d.stake)
            profit = d.stake * profit_factor
            rm.on_trade_closed(profit)
            balance += profit
            trades += 1
        if d is not None and not d.allowed:
            break
    if d is None or d.allowed:
        return False, "daily loss limit never fired"
    if "daily loss limit" not in d.reason:
        return False, f"halted for the wrong reason: {d.reason}"
    if rm.approve_trade(balance).allowed:
        return False, "daily loss halt did not persist"
    lost = BALANCE - balance
    return True, f"daily loss limit fired after losing {lost:.2f} (limit {cfg.daily_loss_limit_pct}% of start balance)"


def check_manual_halt(extra):
    rm, _ = make_manager(Clock())
    rm.approve_trade(BALANCE)
    rm.halt("drill: manual halt")
    d = rm.approve_trade(BALANCE)
    if d.allowed or "drill: manual halt" not in d.reason:
        return False, f"halt() did not block trading (decision: {d})"
    return True, "halt() blocks trading"


def check_kill_file(extra):
    with tempfile.TemporaryDirectory() as tmp:
        stop = Path(tmp) / "STOP"
        rm, _ = make_manager(Clock(), kill_file=stop)
        if not rm.approve_trade(BALANCE).allowed:
            return False, "trading blocked before the kill file existed"
        stop.write_text("stop")
        d = rm.approve_trade(BALANCE)
        if d.allowed or "kill file" not in d.reason:
            return False, f"kill file did not block trading (decision: {d})"
        stop.unlink()
        if not rm.approve_trade(BALANCE).allowed:
            return False, "trading did not resume after the kill file was deleted"
    return True, "STOP file blocks trading and deleting it resumes"


def check_next_day(extra):
    clock = Clock()
    rm, _ = make_manager(clock, daily_loss_limit_pct=100.0, max_trades_per_day=100000)
    count, d, bal = play_losses_until_blocked(rm, BALANCE)
    if d is None:
        return False, "could not trigger a halt to test the reset"
    clock.next_day()
    if not rm.approve_trade(bal).allowed:
        return False, "automatic halt did not clear on the next UTC day"
    return True, "automatic halt clears at the next UTC day"


def check_open_positions(extra):
    rm, cfg = make_manager(Clock())
    d = rm.approve_trade(BALANCE)
    if not d.allowed:
        return False, f"first trade refused: {d.reason}"
    for _ in range(cfg.max_open_positions):
        rm.on_trade_opened(d.stake)
    if rm.approve_trade(BALANCE).allowed:
        return False, "a trade was allowed beyond max_open_positions"
    rm.on_trade_closed(d.stake * 0.1)
    if not rm.approve_trade(BALANCE).allowed:
        return False, "trading did not resume after the open trade closed"
    return True, f"max_open_positions={cfg.max_open_positions} enforced"


CHECKS = [
    ("consecutive-loss halt", check_consecutive),
    ("daily loss limit halt", check_daily_loss),
    ("manual halt()", check_manual_halt),
    ("STOP kill file", check_kill_file),
    ("automatic halt clears next UTC day", check_next_day),
    ("max open positions", check_open_positions),
]


def run():
    extra = {}
    results = []
    for name, fn in CHECKS:
        try:
            ok, detail = fn(extra)
        except Exception as e:  # noqa: BLE001  report, do not crash
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append({"check": name, "passed": ok, "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    passed = all(r["passed"] for r in results)
    out = {
        "passed": passed,
        "ran_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "halted_after_losses": extra.get("halted_after_losses"),
        "checks": results,
        "warnings": extra.get("warnings", []),
    }
    if not passed:
        out["error"] = "; ".join(r["check"] for r in results if not r["passed"]) + " failed"
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "kill_switch_drill.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for w in out["warnings"]:
        print("WARNING:", w)
    print("\nKILL SWITCH DRILL:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(run())