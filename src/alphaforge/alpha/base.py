"""The ``Alpha`` abstraction: a named, causal cross-sectional score."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.data.market_data import MarketData

AlphaExpression = Callable[[MarketData], pd.DataFrame]


@dataclass(frozen=True)
class Alpha:
    """A candidate predictive signal.

    Higher values predict higher *future* returns. Only the cross-sectional ordering is used
    by the evaluation tools, so alphas need no particular scale.

    Attributes:
        name: Unique identifier.
        expression: Causal function ``MarketData -> wide DataFrame``.
        category: Family (momentum, reversal, carry, liquidity, ...).
        description: Economic rationale.
    """

    name: str
    expression: AlphaExpression
    category: str = "misc"
    description: str = ""

    def compute(self, data: MarketData) -> pd.DataFrame:
        """Evaluate the expression; non-finite values and untradable bars become ``NaN``."""
        values = self.expression(data).replace([np.inf, -np.inf], np.nan)
        return values.reindex_like(data.close).where(data.close.notna())
