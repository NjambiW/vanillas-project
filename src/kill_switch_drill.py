"""
Phase 9: kill switch tested on purpose.

Drives the REAL RiskManager (src/risk.py) with your config.py limits and a fake clock, and
the REAL Executor with a fake Deriv connection, and checks every stop mechanism:
  1. consecutive-loss halt fires at exactly max_consecutive_losses and stays on
  2. daily-loss-limit halt fires and stays on
  3. manual halt() (used by the executor on errors) blocks trading
  4. the STOP kill file blocks trading, and trading resumes when the file is deleted
  5. an automatic halt clears at the next UTC day
  6. only max_open_positions trades can be open at once
  7. a halt and the day's loss tally SURVIVE A RESTART (state saved to disk)
  8. the executor places no order while halted or while the STOP file exists

Run from the project root:
    python src/kill_switch_drill.py

Writes reports/kill_switch_drill.json, which go_live_gate.py reads. The result records a
fingerprint of your risk settings: change a setting and the drill must be run again.
"""
import asyncio
import json
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from risk import RiskConfig, RiskManager, settings_fingerprint  # noqa: E402
from staking import make_plan  # noqa: E402

BALANCE = 10_000.0
LOOSE = {"daily_loss_limit_pct": 100.0, "max_trades_per_day": 100000}


class Clock:
    def __init__(self):
        self.t = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t

    def next_day(self):
        self.t = (self.t + timedelta(days=1)).replace(hour=1, minute=0)


def make_manager(clock, state_path=None, **overrides):
    """A RiskManager using YOUR config.py limits. The real STOP file is ignored here (kill_file
    None) so a leftover STOP file cannot disturb the drill; check 4 tests the file on purpose."""
    try:
        import config
        plan_name = config.STAKING_PLAN
        params = ({"fraction": config.RISK_FRACTION} if plan_name == "fixed_fraction"
                  else {"amount": config.FIXED_STAKE} if plan_name == "fixed" else {})
    except Exception:  # noqa: BLE001
        plan_name, params = "fixed_fraction", {"fraction": 0.01}
    overrides.setdefault("kill_file", None)
    cfg = replace(RiskConfig.from_settings(), min_seconds_between_trades=0, **overrides)
    return RiskManager(cfg, make_plan(plan_name, **params), now=clock, state_path=state_path), cfg


def lose_once(rm, balance):
    d = rm.approve_trade(balance)
    if not d.allowed:
        return d, balance
    rm.on_trade_opened(d.stake)
    rm.on_trade_closed(-d.stake)
    return d, balance - d.stake


def play_losses_until_blocked(rm, balance, limit=500):
    """Open/close losing trades until approve_trade refuses. Returns (trades, decision, balance)."""
    count = 0
    while count < limit:
        d, new_balance = lose_once(rm, balance)
        if not d.allowed:
            return count, d, balance
        balance, count = new_balance, count + 1
    return count, None, balance


# ------------------------------------------------------------------ checks
def check_consecutive(extra):
    rm, cfg = make_manager(Clock(), **LOOSE)
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
    return True, f"halted after {count} consecutive losses and stayed halted"


def check_daily_loss(extra):
    rm, cfg = make_manager(Clock(), max_consecutive_losses=100000, max_trades_per_day=100000)
    balance, trades, d = BALANCE, 0, None
    while trades < 5000:
        for profit_factor in (-1.0, -1.0, 0.3):      # two losses then a small win
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
    return True, (f"daily loss limit fired after losing {BALANCE - balance:.2f} "
                  f"(limit {cfg.daily_loss_limit_pct}% of start balance)")


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
    rm, _ = make_manager(clock, **LOOSE)
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


def check_restart_persistence(extra):
    clock = Clock()
    with tempfile.TemporaryDirectory() as tmp:
        # a halt must survive a restart
        state = Path(tmp) / "halt.json"
        rm, _ = make_manager(clock, state_path=state, **LOOSE)
        count, d, bal = play_losses_until_blocked(rm, BALANCE)
        if d is None:
            return False, "could not trigger a halt to test persistence"
        restarted, _ = make_manager(clock, state_path=state, **LOOSE)
        if restarted.approve_trade(bal).allowed:
            return False, "a restart cleared the halt"

        # the day's loss tally must survive a restart too
        state2 = Path(tmp) / "tally.json"
        a, cfg = make_manager(clock, state_path=state2, max_consecutive_losses=100000, max_trades_per_day=100000)
        balance = BALANCE
        limit = BALANCE * cfg.daily_loss_limit_pct / 100
        lost = 0.0
        while True:
            d = a.approve_trade(balance)
            if not d.allowed or lost + d.stake >= limit:
                break                                  # stop just short of the limit
            a.on_trade_opened(d.stake)
            a.on_trade_closed(-d.stake)
            balance -= d.stake
            lost += d.stake
        b, _ = make_manager(clock, state_path=state2, max_consecutive_losses=100000, max_trades_per_day=100000)
        for _ in range(50):                            # keep losing after the "restart"
            d, balance = lose_once(b, balance)
            if not d.allowed:
                return True, "a halt and the day's loss tally both survive a restart"
        return False, "the daily loss tally was reset by a restart"


class _FakeDeriv:
    """Just enough of DerivClient to let the real Executor try to place an order."""

    def __init__(self):
        self.bought = []

    async def get_balance(self):
        return {"balance": BALANCE, "currency": "USD"}

    async def get_allowed_barriers(self, *args):
        return ["+1.10", "+0.00", "-1.10"]

    async def get_proposal(self, ctype, symbol, barrier, duration, unit, stake, currency):
        from pricing import bs_price, years
        kind = "call" if ctype.endswith("CALL") else "put"
        strike = 1000.0 + float(barrier)
        fair = bs_price(kind, 1000.0, strike, years(duration, unit), 1.0)
        return {"id": "p", "ask_price": stake, "display_number_of_contracts": str(stake / (fair * 1.05)),
                "contract_details": {"barrier": f"{strike:.2f}"}, "spot": 1000.0,
                "min_stake": 0.4, "max_stake": 567}

    async def buy(self, proposal_id, max_price):
        self.bought.append(proposal_id)
        return {"contract_id": len(self.bought)}

    async def subscribe(self, payload, callback):
        asyncio.get_running_loop().call_later(
            0.01, callback, {"proposal_open_contract": {"is_sold": 1, "profit": "-1", "status": "lost"}})
        return {"proposal_open_contract": {"is_sold": 0}, "subscription": {"id": "s"}}

    async def forget(self, sub_id):
        pass


def check_executor_blocks_orders(extra):
    try:
        from executor import Executor
        from journal import Journal
        from risk import PremiumGate
        from strategies.base import Signal
    except ImportError as exc:
        return False, f"could not import the executor ({exc})"

    signal = Signal(direction="CALL", reason="drill", atr=2.0, strike_atr=0.5, hold_candles=1)

    class Stub:
        name, hold_candles = "drill", 1

    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            stop = Path(tmp) / "STOP"
            rm, _ = make_manager(Clock(), kill_file=stop)
            client = _FakeDeriv()
            ex = Executor(client, Stub(), rm, PremiumGate(50.0), Journal(), "1HZ100V", 60,
                          dry_run=False, settle_buffer=1.0)
            await ex.handle_signal(signal, 1000.0)
            await ex.wait_idle()
            if len(client.bought) != 1:
                return False, "drill setup problem: the executor did not trade when it was allowed to"
            rm.halt("drill")
            await ex.handle_signal(signal, 1000.0)
            if len(client.bought) != 1:
                return False, "the executor placed an order while the risk manager was halted"
            rm2, _ = make_manager(Clock(), kill_file=stop)
            ex2 = Executor(client, Stub(), rm2, PremiumGate(50.0), Journal(), "1HZ100V", 60,
                           dry_run=False, settle_buffer=1.0)
            stop.write_text("stop")
            await ex2.handle_signal(signal, 1000.0)
            if len(client.bought) != 1:
                return False, "the executor placed an order while the STOP file existed"
            stop.unlink()
            await ex2.handle_signal(signal, 1000.0)
            await ex2.wait_idle()
            if len(client.bought) != 2:
                return False, "the executor did not resume after the STOP file was deleted"
        return True, "no orders while halted or while STOP exists; trading resumes afterwards"

    return asyncio.run(scenario())


CHECKS = [
    ("consecutive-loss halt", check_consecutive),
    ("daily loss limit halt", check_daily_loss),
    ("manual halt()", check_manual_halt),
    ("STOP kill file", check_kill_file),
    ("automatic halt clears next UTC day", check_next_day),
    ("max open positions", check_open_positions),
    ("halt and loss tally survive a restart", check_restart_persistence),
    ("executor places no orders while stopped", check_executor_blocks_orders),
]

NOTES = [
    "The STOP file and halts stop NEW trades. Contracts that are already open keep running to expiry.",
    "These checks test the code paths with a fake clock and a fake connection. Do one live test as "
    "well: start main.py, create the STOP file, and confirm the log says the kill file blocked a trade.",
]


def run(write=True):
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
        "config_fingerprint": settings_fingerprint(),
        "halted_after_losses": extra.get("halted_after_losses"),
        "checks": results,
        "notes": NOTES,
    }
    if not passed:
        out["error"] = "; ".join(r["check"] for r in results if not r["passed"]) + " failed"
    if write:
        (ROOT / "reports").mkdir(exist_ok=True)
        (ROOT / "reports" / "kill_switch_drill.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for note in NOTES:
        print("NOTE:", note)
    print("\nKILL SWITCH DRILL:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(run())