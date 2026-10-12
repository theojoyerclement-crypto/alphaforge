"""Transaction cost models.

Costs are expressed as a fraction of NAV, per asset, for a vector of weight changes
``trades`` executed at a bar's close.

* :class:`LinearCostModel`: exchange fee + half spread, proportional to traded notional.
* :class:`SquareRootImpactModel`: adds market impact ``k * sigma * sqrt(Q / ADV)``, the
  empirical square-root law (Almgren et al. 2005; Toth et al. 2011). Impact grows with
  trade size relative to liquidity, which is what makes capacity finite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from alphaforge.config.schema import CostConfig


class CostModel(ABC):
    """Interface of transaction cost models."""

    @abstractmethod
    def compute(
        self, trades: np.ndarray, nav: float, volatility: np.ndarray, adv: np.ndarray
    ) -> np.ndarray:
        """Cost per asset as a fraction of NAV.

        Args:
            trades: Weight changes (fraction of NAV), one per asset.
            nav: Portfolio value before trading.
            volatility: Per-bar return volatility per asset.
            adv: Average traded dollar volume per bar per asset.
        """


class LinearCostModel(CostModel):
    """Fee plus half spread, both in basis points of traded notional."""

    def __init__(self, fee_bps: float = 4.0, half_spread_bps: float = 1.5) -> None:
        self.fee_bps = fee_bps
        self.half_spread_bps = half_spread_bps

    def compute(
        self, trades: np.ndarray, nav: float, volatility: np.ndarray, adv: np.ndarray
    ) -> np.ndarray:
        return np.abs(trades) * (self.fee_bps + self.half_spread_bps) * 1e-4


class SquareRootImpactModel(LinearCostModel):
    """Linear costs plus square-root market impact.

    ``impact_i = k * sigma_i * sqrt(|trade_i| * NAV / ADV_i)`` per unit traded. When ADV is
    unknown the participation rate is conservatively set to 100%.
    """

    def __init__(
        self, fee_bps: float = 4.0, half_spread_bps: float = 1.5, impact_coefficient: float = 0.1
    ) -> None:
        super().__init__(fee_bps, half_spread_bps)
        self.impact_coefficient = impact_coefficient

    def compute(
        self, trades: np.ndarray, nav: float, volatility: np.ndarray, adv: np.ndarray
    ) -> np.ndarray:
        linear = super().compute(trades, nav, volatility, adv)
        traded = np.abs(trades)
        safe_adv = np.where(np.isfinite(adv) & (adv > 0), adv, np.nan)
        participation = np.where(
            np.isnan(safe_adv), 1.0, traded * nav / np.nan_to_num(safe_adv, nan=1.0)
        )
        sigma = np.nan_to_num(volatility, nan=0.0)
        impact = self.impact_coefficient * sigma * np.sqrt(participation)
        return linear + traded * impact


def build_cost_model(config: CostConfig) -> CostModel:
    """Instantiate the cost model described by ``config``."""
    if config.model == "linear":
        return LinearCostModel(config.fee_bps, config.half_spread_bps)
    return SquareRootImpactModel(config.fee_bps, config.half_spread_bps, config.impact_coefficient)
