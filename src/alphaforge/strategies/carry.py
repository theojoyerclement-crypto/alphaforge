"""Carry family: perpetual funding and basis."""

from __future__ import annotations

import pandas as pd

from alphaforge.data.market_data import MarketData
from alphaforge.features.base import compute_feature
from alphaforge.features.transforms import long_short_quantiles, scale_to_unit_max
from alphaforge.strategies.base import Strategy, register_strategy


@register_strategy
class FundingCarry(Strategy):
    """Cross-sectional carry: short the most expensive perps to hold, long the cheapest.

    The short leg *receives* funding while crowded longs unwind; the long leg pays little or
    is paid. Dollar-neutral, so market beta is mostly hedged.

    Args:
        lookback: Bars of funding (or basis) averaged into the carry estimate.
        quantile: Fraction of the universe per leg.
        source: ``"funding"`` (realised funding) or ``"basis"`` (perp premium over spot).
    """

    name = "funding_carry"
    family = "carry"

    def __init__(self, lookback: int = 21, quantile: float = 0.3, source: str = "funding") -> None:
        if source not in {"funding", "basis"}:
            raise ValueError("source must be 'funding' or 'basis'")
        self.lookback = lookback
        self.quantile = quantile
        self.source = source

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        if self.source == "funding":
            if data.funding is None:
                raise ValueError("FundingCarry(source='funding') requires funding data")
            carry = compute_feature("funding_rate", data, lookback=self.lookback)
        else:
            carry = compute_feature("basis", data, lookback=self.lookback)
        return self.finalize(scale_to_unit_max(long_short_quantiles(-carry, self.quantile)), data)
