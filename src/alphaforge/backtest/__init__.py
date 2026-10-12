"""Backtesting: event-loop engine, cost models, walk-forward and purged cross-validation."""

from alphaforge.backtest.costs import (
    CostModel,
    LinearCostModel,
    SquareRootImpactModel,
    build_cost_model,
)
from alphaforge.backtest.cv import CombinatorialPurgedKFold, PurgedKFold
from alphaforge.backtest.engine import BacktestEngine
from alphaforge.backtest.results import BacktestResult
from alphaforge.backtest.walk_forward import (
    WalkForwardFold,
    WalkForwardResult,
    WalkForwardRunner,
    make_walk_forward_folds,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "CombinatorialPurgedKFold",
    "CostModel",
    "LinearCostModel",
    "PurgedKFold",
    "SquareRootImpactModel",
    "WalkForwardFold",
    "WalkForwardResult",
    "WalkForwardRunner",
    "build_cost_model",
    "make_walk_forward_folds",
]
