"""Regime-aware vanilla options strategy.

Selects between mean_reversion and trend_pullback based on a market regime
classifier. Returns no signal when the regime is uncertain. Only information
available at the decision time is used (all indicators are causal).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from indicators import adx, atr, ema
from strategies.base import Signal, Strategy
from strategies.mean_reversion import MeanReversion
from strategies.trend_pullback import TrendPullback


class Regime(str, Enum):
    RANGING = "ranging"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class RegimeConfig:
    adx_n: int = 14
    ema_fast: int = 20
    ema_slow: int = 50
    atr_n: int = 14
    slope_lookback: int = 5

    # Thresholds (configurable; initial hypotheses)
    adx_trend_min: float = 20.0
    adx_ranging_max: float = 20.0
    min_trend_sep_atr: float = 0.20  # EMA(20)-EMA(50) magnitude >= min_trend_sep_atr * ATR
    min_trend_slope_atr: float = 0.001  # ATR-normalized slope >= threshold
    regime_confirm: int = 1  # require confirmation over this many prior candles (>=1)


class RegimeAware(Strategy):
    name = "regime_aware"

    def __init__(
        self,
        hold_candles: int = 5,
        strike_atr: float = 0.7,
        regime: RegimeConfig | None = None,
        # allow overriding child params if desired
        mr_params: dict | None = None,
        tp_params: dict | None = None,
    ) -> None:
        super().__init__(hold_candles, strike_atr)
        self.regime = regime or RegimeConfig()
        self._mr = MeanReversion(hold_candles=hold_candles, strike_atr=strike_atr, **(mr_params or {}))
        self._tp = TrendPullback(
            fast=self.regime.ema_fast,
            slow=self.regime.ema_slow,
            slope_lookback=self.regime.slope_lookback,
            min_trend_atr=self.regime.min_trend_sep_atr,
            hold_candles=hold_candles,
            strike_atr=strike_atr,
            cooldown_candles=hold_candles,
            **(tp_params or {}),
        )

    @property
    def warmup(self) -> int:
        # Max warmup of components + regime confirmation buffer
        w = max(self._mr.warmup, self._tp.warmup, self.regime.adx_n + self.regime.slope_lookback + 5)
        return w + max(0, self.regime.regime_confirm - 1)

    def describe(self, direction: int) -> str:
        side = "CALL" if direction > 0 else "PUT"
        return f"regime_aware routed {side} via selected child"

    def raw_signals(self, candles: pd.DataFrame) -> pd.Series:
        idx = candles.index
        out = pd.Series(0, index=idx, dtype=int)

        # Get indicators (causal)
        a = adx(candles, self.regime.adx_n)
        atr_now = atr(candles, self.regime.atr_n)
        c = candles["close"]
        efast = ema(c, self.regime.ema_fast)
        eslow = ema(c, self.regime.ema_slow)

        sep_atr = (efast - eslow).abs() / atr_now.replace(0, pd.NA)
        slope = (eslow - eslow.shift(self.regime.slope_lookback)) / atr_now.replace(0, pd.NA)

        regimes = pd.Series(Regime.UNCERTAIN, index=idx, dtype=object)

        # Classify
        for i in range(len(idx)):
            if (
                pd.isna(a.iloc[i])
                or pd.isna(atr_now.iloc[i])
                or pd.isna(efast.iloc[i])
                or pd.isna(eslow.iloc[i])
                or pd.isna(sep_atr.iloc[i])
                or pd.isna(slope.iloc[i])
            ):
                regimes.iloc[i] = Regime.UNCERTAIN
                continue

            adxv = float(a.iloc[i])
            sepatr = float(sep_atr.iloc[i])
            slopev = float(slope.iloc[i])

            # Trending up
            if adxv >= self.regime.adx_trend_min and efast.iloc[i] > eslow.iloc[i] and sepatr >= self.regime.min_trend_sep_atr and slopev >= self.regime.min_trend_slope_atr:
                regimes.iloc[i] = Regime.TRENDING_UP
                continue
            # Trending down
            if adxv >= self.regime.adx_trend_min and efast.iloc[i] < eslow.iloc[i] and sepatr >= self.regime.min_trend_sep_atr and slopev <= -self.regime.min_trend_slope_atr:
                regimes.iloc[i] = Regime.TRENDING_DOWN
                continue
            # Ranging
            if adxv < self.regime.adx_ranging_max and sepatr < self.regime.min_trend_sep_atr:
                regimes.iloc[i] = Regime.RANGING
                continue
            regimes.iloc[i] = Regime.UNCERTAIN

        # Require regime confirmation if requested
        if self.regime.regime_confirm > 1:
            for i in range(len(idx)):
                if regimes.iloc[i] == Regime.UNCERTAIN:
                    continue
                ok = True
                for k in range(1, self.regime.regime_confirm + 1):
                    prev = i - k
                    if prev < 0 or regimes.iloc[prev] != regimes.iloc[i]:
                        ok = False
                        break
                if not ok:
                    regimes.iloc[i] = Regime.UNCERTAIN

        # Route to child strategies
        mr_sig = self._mr.raw_signals(candles)
        tp_sig = self._tp.raw_signals(candles)

        for i in range(len(idx)):
            r = regimes.iloc[i]
            if r == Regime.UNCERTAIN:
                out.iloc[i] = 0
                continue
            if r == Regime.RANGING:
                out.iloc[i] = int(mr_sig.iloc[i]) if pd.notna(mr_sig.iloc[i]) else 0
                continue
            if r in (Regime.TRENDING_UP, Regime.TRENDING_DOWN):
                out.iloc[i] = int(tp_sig.iloc[i]) if pd.notna(tp_sig.iloc[i]) else 0
                continue
            out.iloc[i] = 0

        return out
