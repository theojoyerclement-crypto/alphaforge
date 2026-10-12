"""Backtest result container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alphaforge.analytics.metrics import PerformanceMetrics, compute_metrics


@dataclass
class BacktestResult:
    """Bar-by-bar accounting of a backtest.

    All return series are fractions of NAV at the start of the bar. Identity per bar:
    ``returns = gross_returns + funding - costs`` (up to second-order compounding).
    """

    name: str
    returns: pd.Series  # net returns
    gross_returns: pd.Series  # price PnL only
    funding: pd.Series  # funding PnL (negative = paid)
    costs: pd.Series  # transaction costs (positive = paid)
    turnover: pd.Series  # sum |trade weights|
    positions: pd.DataFrame  # post-trade weights at each close
    equity: pd.Series
    gross_leverage: pd.Series
    net_leverage: pd.Series
    risk_scale: pd.Series  # drawdown-controller multiplier
    periods_per_year: float

    def metrics(self) -> PerformanceMetrics:
        """Performance statistics of the net return stream."""
        return compute_metrics(
            self.returns,
            self.periods_per_year,
            turnover=self.turnover,
            gross_leverage=self.gross_leverage,
            costs=self.costs,
        )

    def to_frame(self) -> pd.DataFrame:
        """Time series of every accounting line, one column each."""
        return pd.DataFrame(
            {
                "returns": self.returns,
                "gross_returns": self.gross_returns,
                "funding": self.funding,
                "costs": self.costs,
                "turnover": self.turnover,
                "equity": self.equity,
                "gross_leverage": self.gross_leverage,
                "net_leverage": self.net_leverage,
                "risk_scale": self.risk_scale,
            }
        )

    def pnl_attribution(self) -> pd.Series:
        """Annualised contribution of price, funding and costs to the net return."""
        ppy = self.periods_per_year
        return pd.Series(
            {
                "price": self.gross_returns.mean() * ppy,
                "funding": self.funding.mean() * ppy,
                "costs": -self.costs.mean() * ppy,
                "net": self.returns.mean() * ppy,
            },
            name=self.name,
        )

    def save(self, directory: str | Path) -> Path:
        """Write time series and positions as CSV under ``directory``."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(out / f"{self.name}_timeseries.csv")
        self.positions.to_csv(out / f"{self.name}_positions.csv")
        return out
