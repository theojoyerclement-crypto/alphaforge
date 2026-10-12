"""Momentum family: time-series (trend following) and cross-sectional."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alphaforge.data.market_data import MarketData
from alphaforge.features.base import compute_feature
from alphaforge.features.transforms import long_short_quantiles, scale_to_unit_max
from alphaforge.strategies.base import Strategy, register_strategy


@register_strategy
class TimeSeriesMomentum(Strategy):
    """Multi-horizon trend following (Moskowitz, Ooi & Pedersen, 2012).

    For each lookback ``L`` the trend t-statistic ``sum(log r, L) / (sigma * sqrt(L))`` is
    squashed by ``tanh`` (or ``sign``); the signal is the average across lookbacks.

    Args:
        lookbacks: Trend horizons in bars.
        vol_lookback: Window for the per-bar volatility normaliser.
        mode: ``"tanh"`` (continuous, lower turnover) or ``"sign"`` (classic binary).
        long_only: Clip shorts (useful for spot-only accounts).
    """

    name = "ts_momentum"
    family = "momentum"

    def __init__(
        self,
        lookbacks: Sequence[int] = (20, 60, 120),
        vol_lookback: int = 60,
        mode: str = "tanh",
        long_only: bool = False,
    ) -> None:
        if mode not in {"tanh", "sign"}:
            raise ValueError("mode must be 'tanh' or 'sign'")
        self.lookbacks = tuple(int(lb) for lb in lookbacks)
        self.vol_lookback = vol_lookback
        self.mode = mode
        self.long_only = long_only

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        lr = data.log_returns()
        vol = lr.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std()
        parts = []
        for lb in self.lookbacks:
            tstat = lr.rolling(lb).sum() / (vol * np.sqrt(lb))
            parts.append(np.tanh(tstat) if self.mode == "tanh" else np.sign(tstat))
        signal = sum(parts) / len(parts)
        if self.long_only:
            signal = signal.clip(lower=0.0)
        return self.finalize(signal, data)


@register_strategy
class CrossSectionalMomentum(Strategy):
    """Long recent winners, short recent losers, dollar-neutral (Jegadeesh & Titman, 1993).

    Args:
        lookback: Formation period.
        skip: Most recent bars excluded (short-term reversal contaminates momentum).
        quantile: Fraction of the universe in each leg.
        vol_adjust: Rank risk-adjusted momentum instead of raw returns.
    """

    name = "xs_momentum"
    family = "momentum"

    def __init__(
        self, lookback: int = 60, skip: int = 1, quantile: float = 0.3, vol_adjust: bool = True
    ) -> None:
        self.lookback = lookback
        self.skip = skip
        self.quantile = quantile
        self.vol_adjust = vol_adjust

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        score = compute_feature("momentum", data, lookback=self.lookback, skip=self.skip)
        if self.vol_adjust:
            score = score / compute_feature("realized_vol", data, lookback=self.lookback)
        weights = long_short_quantiles(score, self.quantile)
        # scale so the most convicted position is +/-1; sizing happens in the portfolio layer
        return self.finalize(scale_to_unit_max(weights), data)
