"""Regime-based strategy: switch between trend and reversion by local market state."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alphaforge.data.market_data import MarketData
from alphaforge.features.base import compute_feature
from alphaforge.strategies.base import Strategy, register_strategy
from alphaforge.strategies.momentum import TimeSeriesMomentum


@register_strategy
class RegimeSwitching(Strategy):
    """Per-asset rule-based regime switch.

    * Trending regime (efficiency ratio above threshold): follow the trend.
    * Choppy regime: fade deviations from the trailing mean (``-tanh(z / 2)``).
    * High-volatility regime (vol above its own trailing quantile): scale everything down.

    All thresholds use trailing windows, so regime labels are causal.
    """

    name = "regime_switching"
    family = "regime"

    def __init__(
        self,
        er_lookback: int = 30,
        er_threshold: float = 0.30,
        trend_lookbacks: Sequence[int] = (20, 60),
        mr_lookback: int = 20,
        vol_lookback: int = 20,
        vol_window: int = 250,
        high_vol_quantile: float = 0.80,
        high_vol_scale: float = 0.5,
    ) -> None:
        self.er_lookback = er_lookback
        self.er_threshold = er_threshold
        self.trend_lookbacks = tuple(trend_lookbacks)
        self.mr_lookback = mr_lookback
        self.vol_lookback = vol_lookback
        self.vol_window = vol_window
        self.high_vol_quantile = high_vol_quantile
        self.high_vol_scale = high_vol_scale

    def regimes(self, data: MarketData) -> pd.DataFrame:
        """Regime labels per asset: 1 trending, 0 choppy (``NaN`` during warm-up)."""
        er = compute_feature("efficiency_ratio", data, lookback=self.er_lookback)
        return (er > self.er_threshold).astype(float).where(er.notna())

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        trend = TimeSeriesMomentum(self.trend_lookbacks).generate_signals(data)
        z = compute_feature("zscore", data, lookback=self.mr_lookback)
        reversion = -np.tanh(z / 2.0)
        regime = self.regimes(data)
        signal = trend.where(regime == 1.0, reversion).where(regime.notna(), 0.0)

        vol = compute_feature("realized_vol", data, lookback=self.vol_lookback)
        threshold = vol.rolling(self.vol_window, min_periods=self.vol_window // 2).quantile(
            self.high_vol_quantile
        )
        signal = signal.where(~(vol > threshold), signal * self.high_vol_scale)
        return self.finalize(signal, data)
