"""
Phase 8: demo forward test review.

Reads the journal (SQLite or CSV), computes the same metrics as the backtest,
compares demo vs backtest, and writes a weakness list.

Run from the project root:
    python src/demo_review.py
    python src/demo_review.py --journal logs/journal.db --backtest reports/backtest_report.json

Backtest report JSON (any of these keys, top level or under
"out_of_sample" / "oos" / "validation" / "test"):
    win_rate (0-1 or 0-100), expectancy, profit_factor, max_drawdown, trades
You can also override from the command line: --bt-win-rate 0.52 --bt-expectancy 0.03 ...

Outputs:
    reports/demo_review.md      human-readable review + weakness list
    reports/demo_metrics.json   machine-readable, used by go_live_gate.py
"""
import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MIN_TRADES = 100

CANDIDATES = [
    ROOT / "logs" / n
    for n in ("journal.db", "journal.sqlite", "journal.csv", "trades.db", "trades.csv")
] + [
    ROOT / "data" / n
    for n in ("journal.db", "journal.sqlite", "journal.csv", "trades.db", "trades.csv")
]

ALIASES = {
    "profit": ["profit", "pnl", "net_pnl", "result_pnl", "profit_loss", "pl"],
    "stake": ["stake", "buy_price", "cost", "ask_price", "ask"],
    "strategy": ["strategy", "strategy_name"],
    "ts": ["ts", "time", "timestamp", "opened_at", "purchase_time", "created_at"],
    "skip": ["skip_reason", "reason_for_skip", "skipped_reason", "reason"],
    "side": ["direction", "side", "contract_type", "signal"],
    "symbol": ["symbol", "underlying"],
}


# ---------------------------------------------------------------- loading
def find_journal(explicit):
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            sys.exit(f"Journal not found: {p}")
        return p
    for p in CANDIDATES:
        if p.exists():
            return p
    sys.exit("No journal found. Pass --journal path/to/journal.(db|csv)")


def pick(keys, field):
    lower = {k.lower(): k for k in keys}
    for a in ALIASES[field]:
        if a in lower:
            return lower[a]
    return None


def load_rows(path):
    if path.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
        con = sqlite3.connect(str(path))
        con.row_factory = sqlite3.Row
        best, best_n = None, -1
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        for t in tables:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
            if pick(cols, "profit") is None:
                continue
            n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            if n > best_n:
                best, best_n = t, n
        if best is None:
            sys.exit(f"No table with a profit/pnl column in {path}. Tables: {tables}")
        rows = [dict(r) for r in con.execute(f'SELECT * FROM "{best}"')]
        con.close()
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_ts(v):
    if v is None or str(v).strip() == "":
        return None
    f = to_float(v)
    if f is not None:
        if f > 1e12:
            f /= 1000.0
        try:
            return datetime.utcfromtimestamp(f)
        except (OverflowError, OSError, ValueError):
            return None
    s = str(v).replace("Z", "").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s[:26], fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------- metrics
def compute_metrics(pnls, stakes=None):
    n = len(pnls)
    if n == 0:
        return {"trades": 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win, gross_loss = sum(wins), -sum(losses)
    cum = peak = max_dd = 0.0
    streak = longest = 0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        if p < 0:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    m = {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / n,
        "total_pnl": sum(pnls),
        "expectancy": sum(pnls) / n,
        "avg_win": gross_win / len(wins) if wins else 0.0,
        "avg_loss": -gross_loss / len(losses) if losses else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
        "max_drawdown": max_dd,
        "longest_losing_streak": longest,
    }
    if stakes and len(stakes) == n and sum(stakes) > 0:
        m["return_per_stake"] = sum(pnls) / sum(stakes)
    return m


def win_rate_ci(p, n):
    if n == 0:
        return (0.0, 1.0)
    half = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)
    return (max(0.0, p - half), min(1.0, p + half))


# ---------------------------------------------------------------- backtest
def _norm_wr(v):
    v = to_float(v)
    if v is None:
        return None
    return v / 100.0 if v > 1.0 else v


def load_backtest(path, overrides):
    bt = {}
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            src = d
            for k in ("out_of_sample", "oos", "validation", "test"):
                if isinstance(d.get(k), dict):
                    src = d[k]
                    break
            bt = {
                "win_rate": _norm_wr(src.get("win_rate")),
                "expectancy": to_float(src.get("expectancy")),
                "profit_factor": to_float(src.get("profit_factor")),
                "max_drawdown": to_float(src.get("max_drawdown")),
                "trades": to_float(src.get("trades")),
            }
    for k, v in overrides.items():
        if v is not None:
            bt[k] = _norm_wr(v) if k == "win_rate" else float(v)
    return {k: v for k, v in bt.items() if v is not None}


# ---------------------------------------------------------------- review
def review(journal=None, backtest=None, overrides=None):
    path = find_journal(journal)
    rows = load_rows(path)
    if not rows:
        sys.exit("Journal is empty.")
    keys = list(rows[0].keys())
    c = {f: pick(keys, f) for f in ALIASES}
    if c["profit"] is None:
        sys.exit(f"No profit column found. Columns: {keys}")

    trades, skips = [], Counter()
    for r in rows:
        p = to_float(r.get(c["profit"]))
        if p is None:
            if c["skip"] and str(r.get(c["skip"]) or "").strip():
                skips[str(r[c["skip"]]).strip()] += 1
            continue
        trades.append(r)

    pnls = [to_float(r[c["profit"]]) for r in trades]
    stakes = [to_float(r.get(c["stake"])) for r in trades] if c["stake"] else None
    if stakes and any(s is None for s in stakes):
        stakes = None
    overall = compute_metrics(pnls, stakes)

    breakdowns = {}
    for field in ("strategy", "side", "symbol"):
        if not c[field]:
            continue
        groups = defaultdict(list)
        for r in trades:
            groups[str(r.get(c[field]))].append(to_float(r[c["profit"]]))
        breakdowns[field] = {k: compute_metrics(v) for k, v in groups.items()}
    if c["ts"]:
        by_hour = defaultdict(list)
        for r in trades:
            t = parse_ts(r.get(c["ts"]))
            if t:
                by_hour[f"{t.hour:02d}:00"].append(to_float(r[c["profit"]]))
        if by_hour:
            breakdowns["hour_utc"] = {k: compute_metrics(v) for k, v in sorted(by_hour.items())}

    bt = load_backtest(backtest, overrides or {})
    flags = []  # (severity, message)
    n = overall["trades"]
    if n < MIN_TRADES:
        flags.append(("INFO", f"Only {n}/{MIN_TRADES} demo trades logged; sample too small for conclusions."))
    if n:
        lo, hi = win_rate_ci(overall["win_rate"], n)
        overall["win_rate_ci95"] = [lo, hi]
        if "win_rate" in bt and not (lo <= bt["win_rate"] <= hi):
            flags.append(("HIGH", f"Backtest win rate {bt['win_rate']:.1%} is outside the demo 95% interval "
                                  f"({lo:.1%} to {hi:.1%}); demo win rate is {overall['win_rate']:.1%}."))
        if "expectancy" in bt:
            e = overall["expectancy"]
            if (e > 0) != (bt["expectancy"] > 0):
                flags.append(("HIGH", f"Expectancy sign differs: demo {e:+.4f} vs backtest {bt['expectancy']:+.4f}."))
            elif bt["expectancy"] > 0 and e < 0.5 * bt["expectancy"]:
                flags.append(("MED", f"Demo expectancy {e:+.4f} is under half of backtest {bt['expectancy']:+.4f}."))
        pf = overall.get("profit_factor")
        if "profit_factor" in bt and pf is not None and pf < 0.7 * bt["profit_factor"]:
            flags.append(("MED", f"Demo profit factor {pf:.2f} vs backtest {bt['profit_factor']:.2f}."))
        if "max_drawdown" in bt and overall["max_drawdown"] > 1.5 * bt["max_drawdown"]:
            flags.append(("MED", f"Demo max drawdown {overall['max_drawdown']:.2f} exceeds 1.5x backtest "
                                 f"({bt['max_drawdown']:.2f})."))
    if not bt:
        flags.append(("INFO", "No backtest numbers supplied; comparison skipped."))

    for field in ("strategy", "side"):
        for k, m in breakdowns.get(field, {}).items():
            if m["trades"] >= 20 and m["expectancy"] < 0:
                flags.append(("MED", f"{field} '{k}': {m['trades']} trades, expectancy {m['expectancy']:+.4f}."))
    total_skips = sum(skips.values())
    if total_skips and n and total_skips > 5 * n:
        flags.append(("INFO", f"{total_skips} skipped signals vs {n} trades; check the gates are not too strict."))

    return {
        "journal": str(path),
        "generated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "overall": overall,
        "backtest": bt,
        "breakdowns": breakdowns,
        "skip_reasons": dict(skips.most_common()),
        "flags": [{"severity": s, "message": m} for s, m in flags],
        "min_trades": MIN_TRADES,
    }


def _fmt(m):
    if not m.get("trades"):
        return "no trades"
    pf = m.get("profit_factor")
    return (f"{m['trades']} trades | win {m['win_rate']:.1%} | exp {m['expectancy']:+.4f} | "
            f"PF {pf:.2f}" if pf is not None else
            f"{m['trades']} trades | win {m['win_rate']:.1%} | exp {m['expectancy']:+.4f} | PF n/a")


def write_reports(res):
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "demo_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    o, bt = res["overall"], res["backtest"]
    L = ["# Demo forward test review", "", f"Generated {res['generated']}  ", f"Journal: `{res['journal']}`", "",
         f"## Overall ({o['trades']}/{res['min_trades']} trades)", ""]
    if o["trades"]:
        L += [f"- Win rate: {o['win_rate']:.1%} (95% interval {o['win_rate_ci95'][0]:.1%} to {o['win_rate_ci95'][1]:.1%})",
              f"- Total P&L: {o['total_pnl']:+.2f}", f"- Expectancy per trade: {o['expectancy']:+.4f}",
              f"- Avg win / avg loss: {o['avg_win']:.3f} / {o['avg_loss']:.3f}",
              f"- Profit factor: {o['profit_factor']:.2f}" if o["profit_factor"] is not None else "- Profit factor: n/a",
              f"- Max drawdown: {o['max_drawdown']:.2f}", f"- Longest losing streak: {o['longest_losing_streak']}"]
        if "return_per_stake" in o:
            L.append(f"- Return per unit staked: {o['return_per_stake']:+.2%}")
    L += ["", "## Backtest reference", ""]
    L += [f"- {k}: {v}" for k, v in bt.items()] or ["- none supplied"]
    L += ["", "## Breakdowns", ""]
    for field, groups in res["breakdowns"].items():
        L.append(f"**{field}**")
        L += [f"- {k}: {_fmt(m)}" for k, m in groups.items()]
        L.append("")
    if res["skip_reasons"]:
        L += ["## Skip reasons", ""] + [f"- {k}: {v}" for k, v in res["skip_reasons"].items()] + [""]
    L += ["## Weakness list (candidates for our joint review)", ""]
    L += [f"- [{f['severity']}] {f['message']}" for f in res["flags"]] or ["- nothing flagged"]
    L += ["", "Rule: change one thing at a time, then re-run the demo and this review.", ""]
    (REPORTS / "demo_review.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal")
    ap.add_argument("--backtest", default="reports/backtest_report.json")
    ap.add_argument("--bt-win-rate", type=float)
    ap.add_argument("--bt-expectancy", type=float)
    ap.add_argument("--bt-profit-factor", type=float)
    ap.add_argument("--bt-max-drawdown", type=float)
    a = ap.parse_args()
    res = review(a.journal, a.backtest, {
        "win_rate": a.bt_win_rate, "expectancy": a.bt_expectancy,
        "profit_factor": a.bt_profit_factor, "max_drawdown": a.bt_max_drawdown})
    write_reports(res)
    o = res["overall"]
    print(f"Demo trades: {o['trades']}/{MIN_TRADES}")
    if o["trades"]:
        print(_fmt(o))
    for f in res["flags"]:
        print(f"[{f['severity']}] {f['message']}")
    print(f"\nWrote {REPORTS / 'demo_review.md'}")


if __name__ == "__main__":
    main()