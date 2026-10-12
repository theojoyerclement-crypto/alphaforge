"""Built-in alpha library: classic, economically motivated crypto signals.

Each alpha is expressed through registered features so it inherits their causality tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alphaforge.alpha.base import Alpha, AlphaExpression
from alphaforge.data.market_data import MarketData
from alphaforge.features.base import compute_feature


def _feat(name: str, sign: float = 1.0, **params: Any) -> AlphaExpression:
    def expression(data: MarketData) -> pd.DataFrame:
        return sign * compute_feature(name, data, **params)

    return expression


def _trend_quality(data: MarketData) -> pd.DataFrame:
    er = compute_feature("efficiency_ratio", data, lookback=30)
    direction = np.sign(compute_feature("momentum", data, lookback=30))
    return er * direction


def _volume_confirmed_momentum(data: MarketData) -> pd.DataFrame:
    mom = compute_feature("vol_adj_momentum", data, lookback=20, vol_lookback=60)
    vz = compute_feature("volume_zscore", data, lookback=20).clip(-2, 2)
    return mom * (1.0 + 0.25 * vz)


def default_alpha_library() -> list[Alpha]:
    """The standard candidate set evaluated by :class:`AlphaResearcher`."""
    return [
        Alpha(
            "mom_20",
            _feat("momentum", lookback=20, skip=1),
            "momentum",
            "Medium-term momentum skipping the last bar (avoids 1-bar reversal).",
        ),
        Alpha(
            "mom_60",
            _feat("momentum", lookback=60, skip=1),
            "momentum",
            "Quarterly momentum: slow-moving trend persistence.",
        ),
        Alpha(
            "vol_adj_mom_60",
            _feat("vol_adj_momentum", lookback=60),
            "momentum",
            "Momentum t-statistic: trend strength per unit of noise.",
        ),
        Alpha(
            "trend_quality_30",
            _trend_quality,
            "momentum",
            "Kaufman efficiency x direction: rewards clean trends over choppy ones.",
        ),
        Alpha(
            "vol_confirmed_mom_20",
            _volume_confirmed_momentum,
            "momentum",
            "Momentum amplified when backed by abnormal volume.",
        ),
        Alpha(
            "reversal_1",
            _feat("returns", sign=-1.0, lookback=1),
            "reversal",
            "1-bar reversal: liquidity provision against short-term overreaction.",
        ),
        Alpha(
            "reversal_5", _feat("returns", sign=-1.0, lookback=5), "reversal", "Weekly reversal."
        ),
        Alpha(
            "rsi_reversal_14",
            _feat("rsi", sign=-1.0, lookback=14),
            "reversal",
            "Overbought/oversold oscillator, faded.",
        ),
        Alpha(
            "dist_to_high_60",
            _feat("distance_to_high", lookback=60),
            "momentum",
            "52-week-high style anchoring: assets near their highs keep outperforming.",
        ),
        Alpha(
            "low_vol_60",
            _feat("realized_vol", sign=-1.0, lookback=60),
            "risk",
            "Low-volatility anomaly: lottery-like assets are overpriced.",
        ),
        Alpha(
            "illiquidity_20",
            _feat("amihud", lookback=20),
            "liquidity",
            "Illiquidity premium: compensation for holding hard-to-trade assets.",
        ),
        Alpha(
            "buy_pressure_10",
            _feat("close_location", lookback=10),
            "microstructure",
            "Closes near bar highs on volume indicate persistent buying.",
        ),
        Alpha(
            "funding_carry_21",
            _feat("funding_rate", sign=-1.0, lookback=21),
            "carry",
            "Short crowded longs (high funding), collect the carry.",
        ),
        Alpha(
            "funding_reversal_90",
            _feat("funding_zscore", sign=-1.0, lookback=90),
            "carry",
            "Funding spikes vs own history mark leverage extremes that unwind.",
        ),
        Alpha(
            "basis_carry_5",
            _feat("basis", sign=-1.0, lookback=5),
            "carry",
            "Rich perp premium over spot predicts underperformance.",
        ),
    ]


def get_alpha(name: str) -> Alpha:
    """Look up an alpha of the default library by name."""
    for alpha in default_alpha_library():
        if alpha.name == name:
            return alpha
    raise KeyError(f"Unknown alpha {name!r}")
