"""Bridge from the alpha research module to the strategy layer."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from alphaforge.alpha.research import AlphaResearcher
from alphaforge.data.market_data import MarketData
from alphaforge.features.transforms import long_short_quantiles, scale_to_unit_max
from alphaforge.strategies.base import Strategy, register_strategy


@register_strategy
class AlphaComposite(Strategy):
    """Trade a causal IC-weighted blend of library alphas, dollar-neutral.

    Weights adapt to each alpha's *realised* trailing IC (see
    :meth:`AlphaResearcher.combine`), so no fitting step and no look-ahead are involved.
    """

    name = "alpha_composite"
    family = "multi_factor"

    def __init__(
        self,
        alphas: Sequence[str] = ("funding_carry_21", "basis_carry_5", "mom_60", "trend_quality_30"),
        ic_lookback: int = 120,
        quantile: float = 0.3,
    ) -> None:
        self.alphas = tuple(alphas)
        self.ic_lookback = ic_lookback
        self.quantile = quantile

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        composite = AlphaResearcher().combine(data, self.alphas, ic_lookback=self.ic_lookback)
        return self.finalize(
            scale_to_unit_max(long_short_quantiles(composite, self.quantile)), data
        )
