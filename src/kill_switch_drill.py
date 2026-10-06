"""
Phase 9: kill switch tested on purpose.

Feeds the real risk module a run of losing trades and confirms that:
  1. trading gets halted,
  2. it stays halted after a subsequent win (no silent reset),
  3. it halts within a sane number of losses.

Run from the project root:
    python src/kill_switch_drill.py

The three ADAPTER functions below guess common names in src/risk.py.
If your risk module uses different names, edit only those three functions.

Writes reports/kill_switch_drill.json, which go_live_gate.py reads.
"""
import importlib
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

LOSS_SIZE = 1.0
MAX_LOSSES_TO_HALT = 50
START_BALANCE = 1000.0


# ------------------------------------------------------------- ADAPTERS
def make_risk():
    mod = importlib.import_module("risk")
    cls = None
    for name in ("RiskManager", "Risk", "RiskEngine", "RiskController", "RiskGuard"):
        cls = getattr(mod, name, None)
        if cls:
            break
    if cls is None:
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if "risk" in name.lower() and obj.__module__ == mod.__name__:
                cls = obj
                break
    if cls is None:
        raise RuntimeError("Could not find a risk class in src/risk.py; edit make_risk().")
    for kwargs in ({}, {"balance": START_BALANCE}, {"starting_balance": START_BALANCE},
                   {"start_balance": START_BALANCE}):
        try:
            return cls(**kwargs)
        except TypeError:
            continue
    raise RuntimeError(f"Could not construct {cls.__name__}; edit make_risk().")


def record(risk, profit):
    for name in ("record_result", "record_trade", "on_result", "register_result",
                 "update", "on_trade_closed", "add_result"):
        fn = getattr(risk, name, None)
        if callable(fn):
            try:
                fn(profit)
                return
            except TypeError:
                continue
    raise RuntimeError("No result-recording method found on the risk object; edit record().")


def is_halted(risk):
    for name in ("killed", "kill_switch", "halted", "is_halted", "is_killed",
                 "trading_halted", "kill_switch_active", "stopped"):
        if hasattr(risk, name):
            v = getattr(risk, name)
            return bool(v() if callable(v) else v)
    for name in ("can_trade", "allow_trade", "allowed", "trading_allowed", "may_trade"):
        fn = getattr(risk, name, None)
        if callable(fn):
            try:
                r = fn()
            except TypeError:
                continue
            if isinstance(r, tuple):
                r = r[0]
            return not bool(r)
    raise RuntimeError("No halted/kill-switch state found on the risk object; edit is_halted().")


# ------------------------------------------------------------- DRILL
def run():
    result = {"passed": False, "ran_at": datetime.utcnow().isoformat(timespec="seconds") + "Z"}
    try:
        risk = make_risk()
        if is_halted(risk):
            raise RuntimeError("Risk object is already halted before the drill started.")
        halted_after = None
        for i in range(1, MAX_LOSSES_TO_HALT + 1):
            record(risk, -LOSS_SIZE)
            if is_halted(risk):
                halted_after = i
                break
        result["halted_after_losses"] = halted_after
        if halted_after is None:
            result["error"] = f"Kill switch never fired in {MAX_LOSSES_TO_HALT} losses."
        else:
            record(risk, +LOSS_SIZE)
            stays = is_halted(risk)
            result["stays_halted_after_win"] = stays
            result["passed"] = bool(stays)
            if not stays:
                result["error"] = "Halt cleared after a winning trade."
    except Exception as e:  # report, do not crash
        result["error"] = f"{type(e).__name__}: {e}"

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "kill_switch_drill.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("\nKILL SWITCH DRILL:", "PASSED" if result["passed"] else "FAILED")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(run())