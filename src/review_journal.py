"""Phase 8 review: what did the bot actually do, and is there any sign of an edge?

Run:  python src/review_journal.py            (all strategies)
      python src/review_journal.py breakout   (one strategy)
"""
import sys

from config import JOURNAL_PATH
from journal import Journal, verdict


def money(x):
    return "n/a" if x is None else f"{x:+.2f}"


def main() -> None:
    if not JOURNAL_PATH.exists():
        raise SystemExit(f"No journal at {JOURNAL_PATH}. Run python src/main.py first.")
    strategy = sys.argv[1] if len(sys.argv) > 1 else None
    j = Journal(JOURNAL_PATH)

    counts = {t: j.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("signals", "skips", "trades")}
    print(f"Journal: {counts['signals']} signals, {counts['skips']} skipped (incl. dry-run), {counts['trades']} real trades\n")

    perf = j.performance(strategy)
    if perf["trades"]:
        print(f"Settled trades:        {perf['trades']}")
        print(f"Win rate:              {perf['win_rate']:.1%}")
        print(f"Total profit:          {money(perf['total_profit'])}")
        print(f"Return per 1.0 staked: {perf['roi']:+.1%}  (after Deriv's markup)")
        print(f"Before the markup:     {perf['gross_roi']:+.1%}  (t = {perf['gross_t_stat']:.2f})")
        print(f"Average markup paid:   {perf['avg_markup_pct']:.1f}%   total cost of markup: {perf['markup_cost']:.2f}")
    print("\n" + verdict(perf))

    print("\nWhy signals were NOT traded (top reasons):")
    for reason, n in j.skip_reasons():
        print(f"  {n:5}  {reason[:110]}")

    print("\nLast 10 trades:")
    for row in j.recent_trades(10):
        ts, ctype, dur, barrier, stake, markup, status, profit = row
        print(f"  {ts[:19]}  {ctype[-4:]:4} {dur:>4} {barrier:>8}  stake {stake:<6} markup {markup or 0:5.1f}%  {status:7} {money(profit)}")
    j.close()


if __name__ == "__main__":
    main()