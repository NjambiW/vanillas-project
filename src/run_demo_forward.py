"""Phase 8: run the REAL bot continuously on your DEMO account and watch how it performs.

Unlike demo_trade_check.py (which buys one random contract to prove the plumbing), this runs
the full pipeline: strategy signal -> risk approval -> price gate -> buy -> settlement -> journal.

Run from the project root:
    python src/run_demo_forward.py                      # until 100 settled trades or Ctrl+C
    python src/run_demo_forward.py --target-trades 150
    python src/run_demo_forward.py --max-hours 12 --report-every 120
    python src/run_demo_forward.py --yes                # skip the YES prompt (unattended restarts)

What it does:
  * asks you to confirm the account is a DEMO account (same idea as demo_trade_check.py)
  * switches DRY_RUN off for this run only (config.py is not edited)
  * prints a performance line every --report-every seconds, read from logs/journal.db
  * stops cleanly (waits for open contracts) at the target, the time limit, or Ctrl+C
  * then writes reports/demo_review.md and reports/demo_metrics.json via demo_review.py

Keep the computer awake and online while it runs. The bot's risk state (daily loss, halts)
lives in memory, so restarting it mid-day resets those counters.
"""
import argparse
import asyncio
import sqlite3
import sys
import time
from pathlib import Path

from config import CLIENTid, JOURNAL_PATH, MAX_MARKUP_PCT, STRATEGY_NAME, SYMBOL
from deriv_client import DerivClient

# Real-money account id prefixes on Deriv. Demo accounts never start with these.
REAL_PREFIXES = ("CR", "MF", "MLT", "MX")


def snapshot(journal_path):
    """Performance numbers from the journal, or None if it can't be read yet."""
    try:
        uri = f"{Path(journal_path).resolve().as_uri()}?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=5)
        try:
            rows = con.execute("SELECT profit, stake, markup_pct FROM trades").fetchall()
            skips = con.execute("SELECT COUNT(*) FROM skips").fetchone()[0]
            signals = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        finally:
            con.close()
    except Exception:  # noqa: BLE001  file or tables not there yet, or briefly locked
        return None
    settled = [r for r in rows if r[0] is not None]
    pnls = [r[0] for r in settled]
    wins = sum(1 for p in pnls if p > 0)
    markups = [r[2] for r in rows if r[2] is not None]
    n = len(pnls)
    return {
        "settled": n,
        "open": len(rows) - n,
        "wins": wins,
        "win_rate": wins / n if n else None,
        "pnl": sum(pnls),
        "expectancy": sum(pnls) / n if n else None,
        "avg_markup": sum(markups) / len(markups) if markups else None,
        "signals": signals,
        "skips": skips,
    }


def format_snapshot(s, target):
    if s is None:
        return "[monitor] waiting for the journal..."
    wr = f"{s['win_rate']:.1%}" if s["win_rate"] is not None else "n/a"
    ex = f"{s['expectancy']:+.3f}" if s["expectancy"] is not None else "n/a"
    mk = f"{s['avg_markup']:.1f}%" if s["avg_markup"] is not None else "n/a"
    return (f"[monitor] settled {s['settled']}/{target} | open {s['open']} | win {wr} | "
            f"P&L {s['pnl']:+.2f} | per trade {ex} | avg markup {mk} | "
            f"signals {s['signals']} | skipped {s['skips']}")


async def confirm_demo(assume_yes):
    client = DerivClient()
    await client.start()
    try:
        bal = await client.get_balance()
    finally:
        await client.close()
    acct = str(CLIENTid)
    print(f"Account {acct}  balance {bal['balance']} {bal['currency']}")
    if acct.upper().startswith(REAL_PREFIXES):
        print("That looks like a REAL-money account id. Refusing to run.")
        return False
    print(f"Strategy {STRATEGY_NAME} on {SYMBOL}, premium gate {MAX_MARKUP_PCT}%.")
    print("This will place real contracts through the API on the account above.")
    print("Only continue if that is your DEMO account.")
    if assume_yes:
        print("--yes given, continuing.")
        return True
    return input("Type YES to continue: ").strip() == "YES"


async def monitor(stop, target, max_seconds, every):
    started = time.monotonic()
    while not stop.is_set():
        await asyncio.sleep(every)
        snap = snapshot(JOURNAL_PATH)
        print(format_snapshot(snap, target), flush=True)
        if snap and snap["settled"] >= target:
            print(f"[monitor] target of {target} settled trades reached, stopping.", flush=True)
            stop.set()
        elif max_seconds and time.monotonic() - started >= max_seconds:
            print("[monitor] time limit reached, stopping.", flush=True)
            stop.set()


async def run(args):
    if not await confirm_demo(args.yes):
        print("Cancelled. Nothing was started.")
        return False

    import main as bot  # imported late so the confirmation above runs first

    bot.DRY_RUN = False  # this run only; config.py is untouched
    stop = asyncio.Event()
    mon = asyncio.create_task(
        monitor(stop, args.target_trades, args.max_hours * 3600 if args.max_hours else 0, args.report_every)
    )
    try:
        await bot.main(stop)
    finally:
        mon.cancel()
    return True


def write_review():
    try:
        import demo_review

        res = demo_review.review(None, "reports/backtest_report.json", {})
        demo_review.write_reports(res)
        print("\n" + format_snapshot(snapshot(JOURNAL_PATH), res["min_trades"]))
        for f in res["flags"]:
            print(f"[{f['severity']}] {f['message']}")
        print(f"\nReview written to {demo_review.REPORTS / 'demo_review.md'}")
    except SystemExit as e:
        print(f"Review skipped: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"Review skipped: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-trades", type=int, default=100)
    ap.add_argument("--max-hours", type=float, default=0, help="0 = no time limit")
    ap.add_argument("--report-every", type=int, default=300, help="seconds between performance lines")
    ap.add_argument("--yes", action="store_true", help="skip the YES prompt")
    args = ap.parse_args()

    ran = False
    try:
        ran = asyncio.run(run(args))
    except KeyboardInterrupt:
        ran = True
        print("\nStopped by Ctrl+C.")
    if ran:
        write_review()
    sys.exit(0)


if __name__ == "__main__":
    main()