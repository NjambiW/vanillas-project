"""
Phase 8: demo forward test review.

Reads the journal (SQLite or CSV), computes the same per-stake metrics as the backtest,
compares demo vs backtest, and writes a weakness list.

Run from the project root:
    python src/demo_review.py
    python src/demo_review.py --journal logs/journal.db --backtest reports/backtest_report.json

The backtest report is written by:  python src/run_backtest.py --strategy NAME --hold N --export
It holds (top level or under "out_of_sample"): roi, win_rate, max_drawdown (in stake units),
avg_markup_pct, profit_factor, trades, strategy, hold_candles, candle_seconds.
Override from the command line with --bt-roi 0.03 --bt-win-rate 0.52 ... if you need to.

IMPORTANT: demo and backtest are compared as RETURN PER 1.0 STAKED, not in money, because the
demo stakes and the backtest stake differ. Comparing money amounts would be meaningless.

Outputs:
    reports/demo_review.md      human-readable review + weakness list
    reports/demo_metrics.json   machine-readable, used by go_live_gate.py

Works with zero trades: it then reports the signals and skip reasons instead.
"""
import argparse
import csv
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MIN_TRADES = 100

CANDIDATES = [
    ROOT / folder / name
    for folder in ("logs", "data")
    for name in ("journal.db", "journal.sqlite", "journal.csv", "trades.db", "trades.csv")
]

ALIASES = {
    "profit": ["profit", "pnl", "net_pnl", "result_pnl", "profit_loss", "pl"],
    "stake": ["stake", "buy_price", "cost", "ask_price", "ask"],
    "markup": ["markup_pct", "markup"],
    "strategy": ["strategy", "strategy_name"],
    "ts": ["ts_open", "ts", "time", "timestamp", "opened_at", "purchase_time", "created_at"],
    "side": ["contract_type", "direction", "side", "signal"],
    "symbol": ["symbol", "underlying"],
    "duration": ["duration"],
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


def is_sqlite(path):
    return path.suffix.lower() in (".db", ".sqlite", ".sqlite3")


def load_trades(path):
    """Returns (rows, columns). Rows may be empty."""
    if is_sqlite(path):
        con = sqlite3.connect(str(path))
        con.row_factory = sqlite3.Row
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        best, best_n = None, -1
        for t in tables:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
            if pick(cols, "profit") is None:
                continue
            n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            if n > best_n:
                best, best_n = t, n
        if best is None:
            con.close()
            sys.exit(f"No table with a profit/pnl column in {path}. Tables: {tables}")
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{best}")')]
        rows = [dict(r) for r in con.execute(f'SELECT * FROM "{best}"')]
        con.close()
        return rows, cols
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def load_skip_reasons(path):
    """Skip reasons from the journal's `skips` table (SQLite only)."""
    if not is_sqlite(path):
        return []
    con = sqlite3.connect(str(path))
    try:
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "skips" not in names:
            return []
        return [r[0] or "" for r in con.execute("SELECT reason FROM skips")]
    finally:
        con.close()


def count_signals(path):
    if not is_sqlite(path):
        return None
    con = sqlite3.connect(str(path))
    try:
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "signals" not in names:
            return None
        return con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    finally:
        con.close()


def to_float(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_ts(v):
    """Handles epoch seconds/ms and ISO strings (with or without timezone). Returns naive UTC."""
    if v is None or str(v).strip() == "":
        return None
    f = to_float(v)
    if f is not None:
        if f > 1e12:
            f /= 1000.0
        try:
            return datetime.fromtimestamp(f, tz=timezone.utc).replace(tzinfo=None)
        except (OverflowError, OSError, ValueError):
            return None
    s = str(v).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def utcnow_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def duration_text(seconds):
    """3600 -> '1h', 300 -> '5m' (same style the journal stores)."""
    seconds = int(seconds)
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


# ---------------------------------------------------------------- metrics
def compute_metrics(pnls, stakes=None):
    """Money-based statistics (informational; not used to compare with the backtest)."""
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


def compute_normalised(pnls, stakes, markups=None):
    """Per-stake statistics, directly comparable with the backtest report.

    roi          average of profit / stake over trades (return per 1.0 staked, after markup)
    roi_se       standard error of that average
    max_drawdown worst peak-to-trough fall of the cumulative per-stake return ("stake units")
    gross_roi    return before the markup, if markups are known (zero means 'no edge')
    """
    n = len(pnls)
    if n == 0 or not stakes or len(stakes) != n or any(s <= 0 for s in stakes):
        return {}
    rois = [p / s for p, s in zip(pnls, stakes)]
    mean = sum(rois) / n
    var = sum((r - mean) ** 2 for r in rois) / (n - 1) if n > 1 else 0.0
    cum = peak = dd = 0.0
    for r in rois:
        cum += r
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    out = {"roi": mean, "roi_se": math.sqrt(var / n) if n > 1 else None, "max_drawdown_units": dd}
    if markups and len(markups) == n and all(m is not None for m in markups):
        gross = [(p + s) * (1 + m / 100) / s - 1 for p, s, m in zip(pnls, stakes, markups)]
        gmean = sum(gross) / n
        gvar = sum((g - gmean) ** 2 for g in gross) / (n - 1) if n > 1 else 0.0
        out["gross_roi"] = gmean
        out["gross_t_stat"] = gmean / math.sqrt(gvar / n) if gvar > 0 and n > 1 else 0.0
        out["avg_markup_pct"] = sum(markups) / n
    return out


def win_rate_ci(p, n):
    if n == 0:
        return (0.0, 1.0)
    half = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)
    return (max(0.0, p - half), min(1.0, p + half))


def summarise_skips(reasons):
    """Group skip reasons (numbers normalised) and pull out gate-rejection markups."""
    grouped = Counter()
    markups = []
    for r in reasons:
        grouped[re.sub(r"\d+(?:\.\d+)?", "#", r).strip() or "(blank)"] += 1
        for m in re.finditer(r"markup\s+([\d.]+)%\s+is above", r):
            markups.append(float(m.group(1)))
    return {
        "total": len(reasons),
        "by_reason": dict(grouped.most_common()),
        "gate_rejections": len(markups),
        "gate_avg_markup": (sum(markups) / len(markups)) if markups else None,
        "gate_min_markup": min(markups) if markups else None,
    }


# ---------------------------------------------------------------- backtest
def _norm_wr(v):
    v = to_float(v)
    if v is None:
        return None
    return v / 100.0 if v > 1.0 else v


NUMERIC_BT_KEYS = ("roi", "gross_roi", "avg_markup_pct", "profit_factor", "max_drawdown", "trades",
                   "p_value", "hold_candles", "candle_seconds", "variants_tested")


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
            bt = {"win_rate": _norm_wr(src.get("win_rate"))}
            for k in NUMERIC_BT_KEYS:
                bt[k] = to_float(src.get(k, d.get(k)))
            # older reports only had "expectancy"; treat it as the per-stake return
            bt["roi"] = bt["roi"] if bt["roi"] is not None else to_float(src.get("expectancy"))
            bt["expectancy"] = bt["roi"]
            for k in ("strategy", "cost_model_source", "generated"):
                if d.get(k):
                    bt[k] = d[k]
            train = d.get("train")
            if isinstance(train, dict):
                bt["train_roi"] = to_float(train.get("roi", train.get("expectancy")))
                bt["train_p_value"] = to_float(train.get("p_value"))
    for k, v in overrides.items():
        if v is not None:
            bt[k] = _norm_wr(v) if k == "win_rate" else float(v)
    if "roi" in overrides and overrides["roi"] is not None:
        bt["expectancy"] = bt["roi"]
    return {k: v for k, v in bt.items() if v is not None}


# ---------------------------------------------------------------- review
def review(journal=None, backtest=None, overrides=None):
    path = find_journal(journal)
    rows, keys = load_trades(path)
    c = {f: pick(keys, f) for f in ALIASES}
    if c["profit"] is None:
        sys.exit(f"No profit column found. Columns: {keys}")

    settled, unsettled = [], 0
    for r in rows:
        if to_float(r.get(c["profit"])) is None:
            unsettled += 1  # still open, or settlement unknown
        else:
            settled.append(r)

    pnls = [to_float(r[c["profit"]]) for r in settled]
    stakes = [to_float(r.get(c["stake"])) for r in settled] if c["stake"] else None
    if stakes and any(s is None for s in stakes):
        stakes = None
    markups = [to_float(r.get(c["markup"])) for r in settled] if c["markup"] else None
    overall = compute_metrics(pnls, stakes)
    overall.update(compute_normalised(pnls, stakes, markups))
    overall["unsettled"] = unsettled

    breakdowns = {}
    for field in ("strategy", "side", "symbol", "duration"):
        if not c[field] or not settled:
            continue
        groups = defaultdict(list)
        for r in settled:
            groups[str(r.get(c[field]))].append(to_float(r[c["profit"]]))
        breakdowns[field] = {k: compute_metrics(v) for k, v in groups.items()}
    if c["ts"] and settled:
        by_hour = defaultdict(list)
        for r in settled:
            t = parse_ts(r.get(c["ts"]))
            if t:
                by_hour[f"{t.hour:02d}:00"].append(to_float(r[c["profit"]]))
        if by_hour:
            breakdowns["hour_utc"] = {k: compute_metrics(v) for k, v in sorted(by_hour.items())}

    skips = summarise_skips(load_skip_reasons(path))
    signals = count_signals(path)

    bt = load_backtest(backtest, overrides or {})
    flags = []  # (severity, message)
    n = overall["trades"]

    if n == 0:
        msg = "No completed trades yet."
        if signals is not None:
            msg += f" {signals} signals logged, {skips['total']} skipped."
        flags.append(("HIGH" if skips["gate_rejections"] else "INFO", msg))
    elif n < MIN_TRADES:
        flags.append(("INFO", f"Only {n}/{MIN_TRADES} demo trades logged; sample too small for conclusions."))

    if skips["gate_rejections"]:
        flags.append((
            "HIGH" if n == 0 else "MED",
            f"{skips['gate_rejections']} signals blocked by the premium gate; average quoted markup "
            f"{skips['gate_avg_markup']:.1f}% (lowest {skips['gate_min_markup']:.1f}%).",
        ))
    if signals and skips["total"] and n:
        if skips["total"] > 5 * n:
            flags.append(("INFO", f"{skips['total']} skipped signals vs {n} trades; check the gates are not too strict."))

    if n:
        lo, hi = win_rate_ci(overall["win_rate"], n)
        overall["win_rate_ci95"] = [lo, hi]

        # --- is the demo the same experiment as the backtest? ---
        demo_strategies = set(breakdowns.get("strategy", {}))
        if bt.get("strategy") and demo_strategies and demo_strategies != {bt["strategy"]}:
            flags.append(("HIGH", f"Demo ran {sorted(demo_strategies)} but the backtest report is for "
                                  f"'{bt['strategy']}'. They cannot be compared."))
        if bt.get("hold_candles") and bt.get("candle_seconds") and "duration" in breakdowns:
            expected = duration_text(bt["hold_candles"] * bt["candle_seconds"])
            seen = set(breakdowns["duration"])
            if seen != {expected}:
                flags.append(("MED", f"Backtest holds {expected}; demo trades used {sorted(seen)}."))

        # --- demo vs backtest, all in per-stake terms ---
        if "win_rate" in bt and not (lo <= bt["win_rate"] <= hi):
            flags.append(("HIGH", f"Backtest win rate {bt['win_rate']:.1%} is outside the demo 95% interval "
                                  f"({lo:.1%} to {hi:.1%}); demo win rate is {overall['win_rate']:.1%}."))
        if "roi" in bt and "roi" in overall:
            d_roi, b_roi, se = overall["roi"], bt["roi"], overall.get("roi_se")
            if (d_roi > 0) != (b_roi > 0):
                flags.append(("HIGH", f"Return per stake changes sign: demo {d_roi:+.1%} vs backtest {b_roi:+.1%}."))
            elif se and abs(d_roi - b_roi) > 2 * se and d_roi < b_roi:
                flags.append(("HIGH", f"Demo return {d_roi:+.1%} (+/-{2 * se:.1%}) is worse than the backtest "
                                      f"{b_roi:+.1%} by more than chance explains."))
            elif b_roi > 0 and d_roi < 0.5 * b_roi:
                flags.append(("MED", f"Demo return {d_roi:+.1%} is under half the backtest {b_roi:+.1%}."))
        pf = overall.get("profit_factor")
        if "profit_factor" in bt and pf is not None and pf < 0.7 * bt["profit_factor"]:
            flags.append(("MED", f"Demo profit factor {pf:.2f} vs backtest {bt['profit_factor']:.2f}."))
        if "max_drawdown" in bt and overall.get("max_drawdown_units") is not None:
            if overall["max_drawdown_units"] > 1.5 * bt["max_drawdown"]:
                flags.append(("MED", f"Demo drawdown {overall['max_drawdown_units']:.2f} stake units exceeds 1.5x "
                                     f"the backtest's {bt['max_drawdown']:.2f}."))
        if "avg_markup_pct" in bt and overall.get("avg_markup_pct") is not None:
            gap = overall["avg_markup_pct"] - bt["avg_markup_pct"]
            if abs(gap) > 5:
                flags.append(("MED", f"Average markup paid {overall['avg_markup_pct']:.1f}% vs {bt['avg_markup_pct']:.1f}% "
                                     "assumed by the backtest; the cost model has drifted, re-run measure_markup.py."))
        if unsettled:
            flags.append(("MED", f"{unsettled} trades have no recorded profit (open or settlement unknown)."))
        for field in ("strategy", "side"):
            for k, m in breakdowns.get(field, {}).items():
                if m["trades"] >= 20 and m["expectancy"] < 0:
                    flags.append(("MED", f"{field} '{k}': {m['trades']} trades, expectancy {m['expectancy']:+.4f}."))
    if not bt:
        flags.append(("INFO", "No backtest numbers supplied; comparison skipped. "
                              "Create them with: python src/run_backtest.py --strategy NAME --hold N --export"))

    return {
        "journal": str(path),
        "generated": utcnow_iso(),
        "overall": overall,
        "signals_logged": signals,
        "backtest": bt,
        "breakdowns": breakdowns,
        "skips": skips,
        "flags": [{"severity": s, "message": m} for s, m in flags],
        "min_trades": MIN_TRADES,
    }


def _fmt(m):
    if not m.get("trades"):
        return "no trades"
    pf = m.get("profit_factor")
    pf_txt = f"{pf:.2f}" if pf is not None else "n/a"
    return f"{m['trades']} trades | win {m['win_rate']:.1%} | exp {m['expectancy']:+.4f} | PF {pf_txt}"


def write_reports(res):
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "demo_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    o, bt, sk = res["overall"], res["backtest"], res["skips"]
    L = ["# Demo forward test review", "", f"Generated {res['generated']}  ", f"Journal: `{res['journal']}`", "",
         f"## Overall ({o['trades']}/{res['min_trades']} settled trades)", ""]
    if o["trades"]:
        L += [f"- Win rate: {o['win_rate']:.1%} (95% interval {o['win_rate_ci95'][0]:.1%} to {o['win_rate_ci95'][1]:.1%})",
              f"- Total P&L: {o['total_pnl']:+.2f}", f"- Expectancy per trade (money): {o['expectancy']:+.4f}",
              f"- Avg win / avg loss: {o['avg_win']:.3f} / {o['avg_loss']:.3f}",
              f"- Profit factor: {o['profit_factor']:.2f}" if o["profit_factor"] is not None else "- Profit factor: n/a",
              f"- Max drawdown (money): {o['max_drawdown']:.2f}", f"- Longest losing streak: {o['longest_losing_streak']}"]
        if "roi" in o:
            se = f" +/- {o['roi_se']:.1%}" if o.get("roi_se") else ""
            L.append(f"- Return per 1.0 staked (after markup): {o['roi']:+.1%}{se}")
            L.append(f"- Drawdown in stake units: {o['max_drawdown_units']:.2f}")
        if "gross_roi" in o:
            L.append(f"- Return before markup: {o['gross_roi']:+.1%} (t = {o['gross_t_stat']:.2f}; "
                     "within +/-2 means no sign of an edge)")
    else:
        L.append("- No settled trades yet.")
    if o.get("unsettled"):
        L.append(f"- Unsettled / unknown: {o['unsettled']}")
    if res["signals_logged"] is not None:
        L.append(f"- Signals logged: {res['signals_logged']}")
    L += ["", "## Backtest reference", ""]
    L += [f"- {k}: {v}" for k, v in bt.items()] or ["- none supplied"]
    L += ["", "## Breakdowns", ""]
    for field, groups in res["breakdowns"].items():
        L.append(f"**{field}**")
        L += [f"- {k}: {_fmt(m)}" for k, m in groups.items()]
        L.append("")
    if sk["total"]:
        L += [f"## Skipped signals ({sk['total']})", ""]
        L += [f"- {k}: {v}" for k, v in sk["by_reason"].items()]
        L.append("")
    L += ["## Weakness list (candidates for our joint review)", ""]
    L += [f"- [{f['severity']}] {f['message']}" for f in res["flags"]] or ["- nothing flagged"]
    L += ["", "Rule: change one thing at a time, then re-run the demo and this review.", ""]
    (REPORTS / "demo_review.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal")
    ap.add_argument("--backtest", default="reports/backtest_report.json")
    ap.add_argument("--bt-win-rate", type=float)
    ap.add_argument("--bt-roi", type=float)
    ap.add_argument("--bt-profit-factor", type=float)
    ap.add_argument("--bt-max-drawdown", type=float, help="in stake units")
    a = ap.parse_args()
    res = review(a.journal, a.backtest, {
        "win_rate": a.bt_win_rate, "roi": a.bt_roi,
        "profit_factor": a.bt_profit_factor, "max_drawdown": a.bt_max_drawdown})
    write_reports(res)
    o = res["overall"]
    print(f"Settled demo trades: {o['trades']}/{MIN_TRADES}")
    if o["trades"]:
        print(_fmt(o))
        if "roi" in o:
            print(f"Return per 1.0 staked: {o['roi']:+.1%}")
    for f in res["flags"]:
        print(f"[{f['severity']}] {f['message']}")
    print(f"\nWrote {REPORTS / 'demo_review.md'}")


if __name__ == "__main__":
    main()