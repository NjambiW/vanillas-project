"""Risk manager: decides whether a trade may happen and how big it may be.

Order of use for every trade the strategy wants:
    1. decision = risk.approve_trade(balance)      -> allowed? what stake?
    2. get a proposal from Deriv for that stake
    3. gate.check(...)                             -> is the price acceptable?
    4. risk.on_trade_opened(stake) after buying
    5. risk.on_trade_closed(profit) when it settles

Every automatic stop lasts until the next UTC day. The kill file (a file called STOP in
the project folder) stops the bot until you delete it.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from markup import MarkupError, markup_row
from staking import StakingPlan, floor_cents


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RiskConfig:
    min_stake: float = 1.0
    max_stake: float = 25.0
    max_risk_pct_per_trade: float = 2.0
    daily_loss_limit_pct: float = 3.0
    max_consecutive_losses: int = 6
    max_trades_per_day: int = 100
    max_open_positions: int = 1
    min_seconds_between_trades: float = 0
    kill_file: Optional[Path] = None

    @classmethod
    def from_settings(cls) -> "RiskConfig":
        import config

        return cls(
            min_stake=config.MIN_STAKE,
            max_stake=config.MAX_STAKE,
            max_risk_pct_per_trade=config.MAX_RISK_PCT_PER_TRADE,
            daily_loss_limit_pct=config.DAILY_LOSS_LIMIT_PCT,
            max_consecutive_losses=config.MAX_CONSECUTIVE_LOSSES,
            max_trades_per_day=config.MAX_TRADES_PER_DAY,
            max_open_positions=config.MAX_OPEN_POSITIONS,
            min_seconds_between_trades=config.MIN_SECONDS_BETWEEN_TRADES,
            kill_file=config.KILL_FILE,
        )


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    stake: float = 0.0
    markup_pct: Optional[float] = None


class RiskManager:
    def __init__(
        self,
        config: RiskConfig,
        staking: StakingPlan,
        now: Callable[[], datetime] = utc_now,
    ):
        self.cfg = config
        self.staking = staking
        self._now = now
        self._day = None
        self._day_start_balance: Optional[float] = None
        self._pnl_today = 0.0
        self._trades_today = 0
        self._consecutive_losses = 0
        self._open_positions = 0
        self._last_trade_time: Optional[datetime] = None
        self._halt_reason: Optional[str] = None

    # ----------------------------------------------------------- bookkeeping
    def _roll_day(self) -> None:
        today = self._now().date()
        if today != self._day:
            self._day = today
            self._day_start_balance = None
            self._pnl_today = 0.0
            self._trades_today = 0
            self._consecutive_losses = 0
            self._halt_reason = None  # automatic halts end with the day

    def on_trade_opened(self, stake: float) -> None:
        self._roll_day()
        self._open_positions += 1
        self._trades_today += 1
        self._last_trade_time = self._now()

    def on_trade_closed(self, profit: float) -> None:
        self._roll_day()
        self._open_positions = max(0, self._open_positions - 1)
        self._pnl_today += profit
        self._consecutive_losses = self._consecutive_losses + 1 if profit < 0 else 0
        self.staking.record(profit)

    def release_position(self) -> None:
        """Free the open-trade slot without recording a result (settlement unknown)."""
        self._open_positions = max(0, self._open_positions - 1)

    # --------------------------------------------------------------- decision
    def approve_trade(self, balance: float) -> Decision:
        self._roll_day()
        cfg = self.cfg

        if cfg.kill_file is not None and Path(cfg.kill_file).exists():
            return Decision(False, f"kill file present ({cfg.kill_file}); delete it to resume")
        if self._day_start_balance is None:
            self._day_start_balance = balance - self._pnl_today
        if self._halt_reason:
            return Decision(False, self._halt_reason)

        loss_limit = self._day_start_balance * cfg.daily_loss_limit_pct / 100
        if self._pnl_today <= -loss_limit:
            self._halt_reason = (
                f"daily loss limit hit ({self._pnl_today:.2f} vs limit -{loss_limit:.2f}); "
                "stopped until the next UTC day"
            )
            return Decision(False, self._halt_reason)
        if self._consecutive_losses >= cfg.max_consecutive_losses:
            self._halt_reason = (
                f"{self._consecutive_losses} losses in a row; stopped until the next UTC day"
            )
            return Decision(False, self._halt_reason)

        if self._open_positions >= cfg.max_open_positions:
            return Decision(False, "a trade is already open")
        if self._trades_today >= cfg.max_trades_per_day:
            return Decision(False, "daily trade limit reached")
        if self._last_trade_time is not None and cfg.min_seconds_between_trades > 0:
            waited = (self._now() - self._last_trade_time).total_seconds()
            if waited < cfg.min_seconds_between_trades:
                return Decision(False, f"cooling down ({waited:.0f}s of {cfg.min_seconds_between_trades:.0f}s)")

        stake = self._size_stake(balance)
        if stake < cfg.min_stake:
            return Decision(
                False,
                f"allowed stake {stake:.2f} is below the minimum {cfg.min_stake} (balance {balance:.2f})",
            )
        return Decision(True, "ok", stake=stake)

    def _size_stake(self, balance: float) -> float:
        cfg = self.cfg
        wanted = self.staking.next_stake(balance)
        cap_pct = balance * cfg.max_risk_pct_per_trade / 100
        stake = min(wanted, cfg.max_stake, cap_pct)
        stake = floor_cents(stake)
        # a plan asking for less than the minimum is raised to it, unless the
        # percent cap forbids even that
        if stake < cfg.min_stake <= min(cfg.max_stake, cap_pct):
            stake = cfg.min_stake
        return stake

    # ---------------------------------------------------------------- control
    def halt(self, reason: str) -> None:
        """Stop trading until the next UTC day (call this from error handlers too)."""
        self._roll_day()  # make sure today's reset has happened, so it cannot wipe this halt
        self._halt_reason = reason

    @property
    def pnl_today(self) -> float:
        return self._pnl_today


class PremiumGate:
    """Rejects trades whose price is too far above the Black-Scholes fair value.

    Fails closed: if we cannot work out the markup, the trade is rejected.
    """

    def __init__(self, max_markup_pct: float):
        self.max_markup_pct = max_markup_pct

    def check(
        self,
        contract_type: str,
        proposal: dict,
        spot: float,
        barrier: str,
        duration: int,
        unit: str,
        symbol: str,
    ) -> Decision:
        try:
            row = markup_row(contract_type, proposal, spot, barrier, duration, unit, symbol)
        except MarkupError as exc:
            return Decision(False, f"cannot verify the price, so not trading ({exc})")
        markup = row["markup_pct"]
        if markup != markup:  # NaN: fair value was essentially zero
            return Decision(False, "fair value too small to judge the price", markup_pct=None)
        if markup > self.max_markup_pct:
            return Decision(
                False, f"markup {markup:.1f}% is above the {self.max_markup_pct}% limit", markup_pct=markup
            )
        return Decision(True, "price acceptable", markup_pct=markup)