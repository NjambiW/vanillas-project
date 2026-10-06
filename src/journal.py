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