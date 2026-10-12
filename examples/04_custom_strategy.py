"""Extending AlphaForge: register a new strategy and run it through the full stack.

The strategy below goes long assets whose funding is negative (shorts are paying to hold)
and whose trend is up: a "squeeze" setup. Once registered it is usable from YAML configs
exactly like the built-ins (``- name: funding_squeeze``).

    python examples/04_custom_strategy.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.config import ExperimentConfig
from alphaforge.data import MarketData
from alphaforge.features import compute_feature
from alphaforge.research import run_experiment
from alphaforge.strategies import Strategy, register_strategy


@register_strategy
class FundingSqueeze(Strategy):
    """Long when funding is negative and the medium-term trend is positive."""

    name = "funding_squeeze"
    family = "carry"

    def __init__(self, funding_lookback: int = 7, trend_lookback: int = 30) -> None:
        self.funding_lookback = funding_lookback
        self.trend_lookback = trend_lookback

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        funding = compute_feature("funding_rate", data, lookback=self.funding_lookback)
        trend = compute_feature("vol_adj_momentum", data, lookback=self.trend_lookback)
        signal = ((funding < 0) & (trend > 0)).astype(float) * np.tanh(trend)
        return self.finalize(signal, data)


def main() -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "custom_strategy_demo",
            "data": {
                "symbols": ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX"],
                "synthetic": {"n_bars": 1500},
            },
            "strategies": [{"name": "funding_squeeze"}, {"name": "funding_carry"}],
            "backtest": {"start": "2020-07-01"},
            "report": {"save": False},
        }
    )
    result = run_experiment(config)
    print(result.summary().round(3))


if __name__ == "__main__":
    main()
