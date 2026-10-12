"""Mean-reversion family: time-series z-score with hysteresis, cross-sectional reversal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.market_data import MarketData
from alphaforge.features.base import compute_feature
from alphaforge.features.transforms import long_short_quantiles, scale_to_unit_max
from alphaforge.strategies.base import Strategy, register_strategy


def hysteresis_positions(z: np.ndarray, entry: float, exit_: float) -> np.ndarray:
    """Stateful band strategy: enter beyond ``+/-entry``, flatten inside ``+/-exit``.

    Fades the deviation (short when ``z > entry``). Positions persist between the bands,
    which cuts turnover compared to a memoryless threshold rule.
    """
    n, k = z.shape
    out = np.zeros((n, k))
    pos = np.zeros(k)
    for t in range(n):
        zt = z[t]
        valid = np.isfinite(zt)
        pos = np.where(valid & (zt > entry), -1.0, pos)
        pos = np.where(valid & (zt < -entry), 1.0, pos)
        pos = np.where(valid & (np.abs(zt) < exit_), 0.0, pos)
        pos = np.where(valid, pos, 0.0)
        out[t] = pos
    return out


@register_strategy
class ZScoreMeanReversion(Strategy):
    """Fade stretched prices back toward their trailing mean.

    A trend filter (Kaufman efficiency ratio) disables the strategy in clean trends, the main
    way naive mean reversion blows up in crypto.

    Args:
        lookback: Window of the price z-score.
        entry_z: Entry threshold.
        exit_z: Exit threshold (``< entry_z``).
        max_efficiency: Only trade when efficiency ratio is below this (``None`` = always).
        er_lookback: Efficiency ratio window.
    """

    name = "zscore_reversion"
    family = "mean_reversion"

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = 1.5,
        exit_z: float = 0.25,
        max_efficiency: float | None = 0.35,
        er_lookback: int = 20,
    ) -> None:
        if exit_z >= entry_z:
            raise ValueError("exit_z must be smaller than entry_z")
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_efficiency = max_efficiency
        self.er_lookback = er_lookback

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        z = compute_feature("zscore", data, lookback=self.lookback)
        pos = pd.DataFrame(
            hysteresis_positions(z.to_numpy(), self.entry_z, self.exit_z),
            index=z.index,
            columns=z.columns,
        )
        if self.max_efficiency is not None:
            er = compute_feature("efficiency_ratio", data, lookback=self.er_lookback)
            pos = pos.where(er < self.max_efficiency, 0.0)
        return self.finalize(pos, data)


@register_strategy
class ShortTermReversal(Strategy):
    """Cross-sectional short-term reversal: buy the recent losers, sell the recent winners.

    Profits from liquidity provision against overreaction; turnover is high, so this sleeve
    is the most sensitive to the cost model.
    """

    name = "st_reversal"
    family = "mean_reversion"

    def __init__(self, lookback: int = 1, quantile: float = 0.3) -> None:
        self.lookback = lookback
        self.quantile = quantile

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        score = -compute_feature("returns", data, lookback=self.lookback)
        return self.finalize(scale_to_unit_max(long_short_quantiles(score, self.quantile)), data)
