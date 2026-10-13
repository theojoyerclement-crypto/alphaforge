"""Performance analytics and reporting.

``metrics`` and ``drawdown`` are imported eagerly; plotting/reporting (matplotlib) is imported
from its submodules on demand.
"""

from alphaforge.analytics.drawdown import drawdown_table
from alphaforge.analytics.inference import SharpeInterval, sharpe_difference, sharpe_interval
from alphaforge.analytics.metrics import (
    PerformanceMetrics,
    annualized_volatility,
    cagr,
    calmar_ratio,
    compute_metrics,
    conditional_value_at_risk,
    drawdown_series,
    equity_curve,
    hit_rate,
    max_drawdown,
    max_drawdown_duration,
    monthly_returns,
    profit_factor,
    rolling_sharpe,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    value_at_risk,
)

__all__ = [
    "PerformanceMetrics",
    "SharpeInterval",
    "annualized_volatility",
    "cagr",
    "calmar_ratio",
    "compute_metrics",
    "conditional_value_at_risk",
    "drawdown_series",
    "drawdown_table",
    "equity_curve",
    "hit_rate",
    "max_drawdown",
    "max_drawdown_duration",
    "monthly_returns",
    "profit_factor",
    "rolling_sharpe",
    "sharpe_difference",
    "sharpe_interval",
    "sharpe_ratio",
    "sortino_ratio",
    "tail_ratio",
    "value_at_risk",
]
