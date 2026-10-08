"""Trade journal in a local SQLite file (no extra install needed).

Records every signal, every reason a signal was NOT traded, and every trade with its
price, markup and result. This is how we later find out what is really working.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY, ts TEXT, strategy TEXT, direction TEXT,
    reason TEXT, atr REAL, price REAL);
CREATE TABLE IF NOT EXISTS skips (
    id INTEGER PRIMARY KEY, ts TEXT, strategy TEXT, direction TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY, ts_open TEXT, ts_close TEXT, strategy TEXT,
    contract_id TEXT, contract_type TEXT, symbol TEXT, duration TEXT,
    barrier TEXT, strike REAL, spot REAL, stake REAL, payout_per_point REAL,
    markup_pct REAL, status TEXT, profit REAL);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _t_stat(xs: list) -> Optional[float]:
    """One-sample t statistic of `xs` against zero, or None when it is undefined."""
    n = len(xs)
    if n < 2:
        return None
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    if var <= 0:
        return None
    return mean / (var ** 0.5) / (n ** 0.5)


# Trades before we will call an edge an edge, and before a t statistic means much.
MIN_TRADES_FOR_AN_EDGE = 100


def verdict(perf: dict) -> str:
    """Plain reading of a `performance()` dict -- no praise for a small sample."""
    trades = perf.get("trades") or 0
    roi = perf.get("roi")
    t_stat = perf.get("gross_t_stat")
    if trades == 0:
        return "No settled trades yet: there is nothing to judge."
    head = (f"{trades} settled trades, ROI {roi:+.1%} after markup, "
            f"win rate {perf['win_rate']:.1%}. ")
    if trades < MIN_TRADES_FOR_AN_EDGE:
        return (head + f"Fewer than {MIN_TRADES_FOR_AN_EDGE} trades, so this is a "
                "smoke test, not evidence -- in either direction.")
    if t_stat is None:
        return head + "Not enough variation to test whether the result is skill."
    if roi > 0 and t_stat >= 2:
        return head + f"gross t = {t_stat:.2f} (>= 2): the edge survives pricing."
    if roi > 0:
        return head + (f"gross t = {t_stat:.2f} (< 2): positive, but a run this small "
                       "is consistent with luck.")
    return head + f"gross t = {t_stat:.2f}: no edge shown even before the markup."


class Journal:
    def __init__(self, path: Union[str, Path] = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # ---------------------------------------------------------------- writes
    def log_signal(self, strategy: str, direction: str, reason: str, atr: float, price: float) -> None:
        self.db.execute(
            "INSERT INTO signals (ts, strategy, direction, reason, atr, price) VALUES (?,?,?,?,?,?)",
            (_now(), strategy, direction, reason, atr, price),
        )
        self.db.commit()

    def log_skip(self, strategy: str, direction: str, reason: str) -> None:
        self.db.execute(
            "INSERT INTO skips (ts, strategy, direction, reason) VALUES (?,?,?,?)",
            (_now(), strategy, direction, reason),
        )
        self.db.commit()

    def open_trade(self, **f) -> int:
        cur = self.db.execute(
            """INSERT INTO trades (ts_open, strategy, contract_id, contract_type, symbol, duration,
               barrier, strike, spot, stake, payout_per_point, markup_pct, status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?, 'open')""",
            (
                _now(), f.get("strategy"), str(f.get("contract_id")), f.get("contract_type"),
                f.get("symbol"), f.get("duration"), f.get("barrier"), f.get("strike"),
                f.get("spot"), f.get("stake"), f.get("payout_per_point"), f.get("markup_pct"),
            ),
        )
        self.db.commit()
        return cur.lastrowid

    def close_trade(self, trade_id: int, profit: Optional[float], status: str) -> None:
        self.db.execute(
            "UPDATE trades SET ts_close=?, profit=?, status=? WHERE id=?",
            (_now(), profit, status, trade_id),
        )
        self.db.commit()

    # ----------------------------------------------------------------- reads
    def summary(self) -> dict:
        rows = self.db.execute(
            "SELECT status, COUNT(*), COALESCE(SUM(profit),0) FROM trades GROUP BY status"
        ).fetchall()
        by = {status: (n, total) for status, n, total in rows}
        wins, losses = by.get("won", (0, 0.0)), by.get("lost", (0, 0.0))
        closed = wins[0] + losses[0]
        total_profit = wins[1] + losses[1]
        return {
            "trades_closed": closed,
            "open": by.get("open", (0, 0))[0],
            "unknown": by.get("unknown", (0, 0))[0],
            "wins": wins[0],
            "losses": losses[0],
            "win_rate": wins[0] / closed if closed else None,
            "total_profit": total_profit,
            "avg_win": wins[1] / wins[0] if wins[0] else None,
            "avg_loss": losses[1] / losses[0] if losses[0] else None,
            "expectancy": total_profit / closed if closed else None,
        }

    def performance(self, strategy: Optional[str] = None) -> dict:
        """Settled-trade performance, net of Deriv's markup.

        The markup is charged on top of fair value at entry, so `markup_cost` is what
        that charge was worth in money: stake x markup%. Adding it back gives the
        `gross_*` figures -- what the same entries would have returned had the quote
        been free -- which is how we separate a real edge from a pricing question.
        """
        sql = ("SELECT stake, markup_pct, profit FROM trades "
               "WHERE status IN ('won','lost') AND profit IS NOT NULL")
        params: tuple = ()
        if strategy:
            sql += " AND strategy=?"
            params = (strategy,)
        rows = self.db.execute(sql, params).fetchall()
        if not rows:
            return {"trades": 0, "win_rate": None, "total_profit": 0.0, "roi": None,
                    "gross_roi": None, "gross_t_stat": None, "avg_markup_pct": None,
                    "markup_cost": 0.0}

        stakes = [float(s) for s, _, _ in rows]
        markups = [(m if m is not None else 0.0) for _, m, _ in rows]
        pnls = [float(p) for _, _, p in rows]
        wins = [p for p in pnls if p > 0]

        # markup_pct = ask/fair - 1, and the whole ask is the stake, so the part of the
        # stake that was markup is ask - fair = ask * m/(100+m). Using ask*m here would
        # exceed the stake itself on the deep-OTM rows, where m runs past 150%.
        costs = [s * m / (100.0 + m) for s, m in zip(stakes, markups)]
        total_stake = sum(stakes)
        markup_cost = sum(costs)
        total_profit = sum(pnls)
        gross_rois = [(p + c) / (s - c)
                      for p, s, c in zip(pnls, stakes, costs) if s - c > 0]
        return {
            "trades": len(rows),
            "win_rate": len(wins) / len(rows),
            "total_profit": total_profit,
            "roi": total_profit / total_stake if total_stake else None,
            "gross_roi": (sum(gross_rois) / len(gross_rois)) if gross_rois else None,
            "gross_t_stat": _t_stat(gross_rois),
            "avg_markup_pct": sum(markups) / len(markups),
            "markup_cost": markup_cost,
        }

    def skip_reasons(self) -> list:
        return self.db.execute(
            "SELECT reason, COUNT(*) AS n FROM skips GROUP BY reason ORDER BY n DESC LIMIT 10"
        ).fetchall()

    def recent_trades(self, n: int = 10) -> list:
        return self.db.execute(
            "SELECT ts_open, contract_type, duration, barrier, stake, markup_pct, status, profit "
            "FROM trades ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()