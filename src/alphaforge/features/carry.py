"""Derivatives carry features: perpetual funding and basis.

Funding is the dominant carry of crypto perps. Persistently high funding signals crowded
leveraged longs, and holding a short there *earns* that funding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.market_data import MarketData
from alphaforge.features.base import register_feature


def _missing(data: MarketData) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)


@register_feature("funding_rate", "carry", "Annualised trailing mean funding (cost to longs)")
def funding_rate(data: MarketData, lookback: int = 21) -> pd.DataFrame:
    if data.funding is None:
        return _missing(data)
    mean = data.funding.rolling(lookback, min_periods=max(1, lookback // 2)).mean()
    return mean * data.periods_per_year


@register_feature("funding_zscore", "carry", "Z-score of smoothed funding vs its own history")
def funding_zscore(data: MarketData, lookback: int = 90, smooth: int = 3) -> pd.DataFrame:
    if data.funding is None:
        return _missing(data)
    smoothed = data.funding.rolling(smooth).mean()
    roll = smoothed.rolling(lookback, min_periods=max(10, lookback // 3))
    return (smoothed - roll.mean()) / roll.std().replace(0.0, np.nan)


@register_feature("basis", "carry", "Trailing mean annualised perp-spot basis")
def basis(data: MarketData, lookback: int = 5) -> pd.DataFrame:
    panel = data.extras.get("basis")
    if panel is None:
        return _missing(data)
    return panel.rolling(lookback, min_periods=1).mean()
