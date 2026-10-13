"""Portfolio construction: strategy signals -> sized sleeves -> allocated, risk-targeted book.

Pipeline::

    signals[s]  --inverse vol + sleeve vol target-->  sleeve_weights[s]
    sleeve_weights  --causal sleeve returns-->  StrategyAllocator  -->  allocation[t, s]
    sum_s allocation[t, s] * sleeve_weights[s]  --portfolio vol target + limits-->  targets
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.config.schema import PortfolioConfig, UniverseConfig
from alphaforge.data.market_data import MarketData
from alphaforge.data.universe import universe_mask
from alphaforge.features.transforms import normalize_gross
from alphaforge.portfolio.allocation import StrategyAllocator
from alphaforge.portfolio.risk import PositionLimits, apply_position_limits
from alphaforge.portfolio.sizing import asset_volatility, inverse_volatility, volatility_target


@dataclass
class PortfolioTargets:
    """Output of :meth:`PortfolioConstructor.build`."""

    weights: pd.DataFrame  # final target weights (fraction of NAV) per asset
    sleeve_weights: dict[str, pd.DataFrame]  # stand-alone sized book of each strategy
    allocation: pd.DataFrame  # strategy allocation weights (rows sum to 1)
    sleeve_returns: pd.DataFrame  # pre-cost sleeve returns used by the allocator
    vol_scale: pd.Series  # portfolio-level volatility-targeting multiplier


class PortfolioConstructor:
    """Turns raw strategy signals into a risk-managed target book."""

    def __init__(
        self, config: PortfolioConfig | None = None, universe: UniverseConfig | None = None
    ) -> None:
        self.config = config or PortfolioConfig()
        self.universe = universe
        self.limits = PositionLimits(
            self.config.max_position, self.config.max_gross_leverage, self.config.max_net_leverage
        )

    def tradable(self, data: MarketData) -> pd.DataFrame:
        """Where positions may be held: listed bars, gated by the point-in-time universe."""
        if self.universe is None:
            return data.close.notna()
        return universe_mask(data, self.universe)

    def size_sleeve(
        self, signals: pd.DataFrame, data: MarketData, tradable: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Risk-balance a strategy's signals across assets and scale to its vol target.

        Signals outside ``tradable`` (default :meth:`tradable`) are zeroed *before* sizing, so
        the sleeve's risk budget is spread over members only. Strategies still compute their
        signals on the full candidate set, which keeps lookback history for symbols that join.
        """
        cfg = self.config
        returns = data.returns()
        mask = self.tradable(data) if tradable is None else tradable
        sig = signals.reindex_like(data.close).fillna(0.0).clip(-1.0, 1.0)
        sig = sig.where(mask, 0.0)
        vol = asset_volatility(returns, cfg.vol_halflife, data.periods_per_year)
        raw = inverse_volatility(sig, vol)
        if cfg.sleeve_target_vol is None:
            return normalize_gross(raw)
        sized, _ = volatility_target(
            raw,
            returns,
            cfg.sleeve_target_vol,
            cfg.vol_halflife,
            data.periods_per_year,
            max_leverage=cfg.max_gross_leverage,
        )
        return sized

    def build(self, signals: Mapping[str, pd.DataFrame], data: MarketData) -> PortfolioTargets:
        """Combine strategy signals into final target weights."""
        if not signals:
            raise ValueError("No strategy signals to combine")
        cfg = self.config
        returns = data.returns().fillna(0.0)
        tradable = self.tradable(data)
        sleeves = {name: self.size_sleeve(sig, data, tradable) for name, sig in signals.items()}
        sleeve_returns = pd.DataFrame(
            {name: (w.shift(1) * returns).sum(axis=1) for name, w in sleeves.items()}
        )

        if len(sleeves) == 1:
            allocation = pd.DataFrame(1.0, index=data.index, columns=list(sleeves))
        else:
            allocation = StrategyAllocator(
                cfg.allocation,
                cfg.allocation_lookback,
                cfg.rebalance_every,
                cfg.correlation_threshold,
                cfg.allocation_min_history,
            ).allocate(sleeve_returns)

        combined = pd.DataFrame(0.0, index=data.index, columns=data.close.columns)
        for name, sleeve in sleeves.items():
            combined = combined + sleeve.mul(allocation[name], axis=0)
        if cfg.target_vol is not None:
            combined, scale = volatility_target(
                combined,
                data.returns(),
                cfg.target_vol,
                cfg.vol_halflife,
                data.periods_per_year,
                max_leverage=cfg.max_gross_leverage,
            )
        else:
            scale = pd.Series(1.0, index=data.index, name="vol_target_scale")
        weights = apply_position_limits(combined, self.limits)
        weights = weights.where(tradable, 0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return PortfolioTargets(weights, sleeves, allocation, sleeve_returns, scale)
