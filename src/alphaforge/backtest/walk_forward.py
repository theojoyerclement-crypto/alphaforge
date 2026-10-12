"""Walk-forward evaluation: refit on the past, trade the next unseen window, repeat.

Only out-of-sample signals are ever traded. Each fold builds *fresh* strategy instances,
so no fitted state leaks from one fold to the next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphaforge.analytics.metrics import compute_metrics
from alphaforge.backtest.engine import BacktestEngine
from alphaforge.backtest.results import BacktestResult
from alphaforge.data.market_data import MarketData
from alphaforge.portfolio.construction import PortfolioConstructor, PortfolioTargets
from alphaforge.utils.logging import get_logger

if TYPE_CHECKING:  # runtime import would create a cycle: strategies -> ml -> backtest
    from alphaforge.strategies.base import StrategyFactory

logger = get_logger(__name__)


@dataclass(frozen=True)
class WalkForwardFold:
    """Positional bar ranges (end-exclusive) of one fold."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def make_walk_forward_folds(
    n_bars: int,
    train_bars: int,
    test_bars: int,
    expanding: bool = True,
    embargo_bars: int = 0,
) -> list[WalkForwardFold]:
    """Sequential train/test folds; test windows tile the data after the first training set.

    The embargo leaves a gap between the end of training and the start of testing so labels
    computed near the end of the training window cannot overlap the test period.
    """
    if train_bars + embargo_bars >= n_bars:
        raise ValueError("Not enough bars for a single walk-forward fold")
    folds: list[WalkForwardFold] = []
    test_start = train_bars + embargo_bars
    while test_start < n_bars:
        test_end = min(test_start + test_bars, n_bars)
        train_end = test_start - embargo_bars
        train_start = 0 if expanding else max(0, train_end - train_bars)
        folds.append(WalkForwardFold(len(folds), train_start, train_end, test_start, test_end))
        test_start = test_end
    return folds


@dataclass
class WalkForwardResult:
    """Out-of-sample backtest plus per-fold diagnostics."""

    backtest: BacktestResult
    targets: PortfolioTargets
    folds: list[WalkForwardFold]
    fold_metrics: pd.DataFrame
    oos_signals: dict[str, pd.DataFrame]
    oos_start: pd.Timestamp


class WalkForwardRunner:
    """Orchestrates fit -> predict -> construct -> simulate across folds.

    Args:
        strategy_factory: Returns a fresh ``{label: Strategy}`` dict on every call.
        constructor: Portfolio constructor.
        engine: Backtest engine.
        train_bars, test_bars, expanding, embargo_bars: Fold geometry.
    """

    def __init__(
        self,
        strategy_factory: StrategyFactory,
        constructor: PortfolioConstructor,
        engine: BacktestEngine,
        train_bars: int = 500,
        test_bars: int = 120,
        expanding: bool = True,
        embargo_bars: int = 5,
    ) -> None:
        self.strategy_factory = strategy_factory
        self.constructor = constructor
        self.engine = engine
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.expanding = expanding
        self.embargo_bars = embargo_bars

    def run(self, data: MarketData) -> WalkForwardResult:
        folds = make_walk_forward_folds(
            data.n_bars, self.train_bars, self.test_bars, self.expanding, self.embargo_bars
        )
        shape = (data.n_bars, len(data.symbols))
        oos: dict[str, np.ndarray] = {}
        for fold in folds:
            strategies = self.strategy_factory()
            train = data.iloc(fold.train_start, fold.train_end)
            history = data.iloc(0, fold.test_end)  # causal features need the full past
            for label, strategy in strategies.items():
                if strategy.requires_fit:
                    strategy.fit(train)
                signals = strategy.generate_signals(history).to_numpy()
                buf = oos.setdefault(label, np.zeros(shape))
                buf[fold.test_start : fold.test_end] = signals[fold.test_start : fold.test_end]
            logger.info(
                "Fold %d: train %s..%s, test %s..%s",
                fold.fold,
                data.index[fold.train_start].date(),
                data.index[fold.train_end - 1].date(),
                data.index[fold.test_start].date(),
                data.index[fold.test_end - 1].date(),
            )

        oos_signals = {
            label: pd.DataFrame(buf, index=data.index, columns=data.close.columns)
            for label, buf in oos.items()
        }
        targets = self.constructor.build(oos_signals, data)
        oos_start = data.index[folds[0].test_start]
        result = self.engine.run(targets.weights, data, start=oos_start, name="walk_forward")

        rows = []
        for fold in folds:
            rets = result.returns.loc[data.index[fold.test_start] : data.index[fold.test_end - 1]]
            m = compute_metrics(rets, data.periods_per_year)
            rows.append(
                {
                    "fold": fold.fold,
                    "test_start": data.index[fold.test_start],
                    "test_end": data.index[fold.test_end - 1],
                    "return": m.total_return,
                    "sharpe": m.sharpe,
                    "max_drawdown": m.max_drawdown,
                }
            )
        return WalkForwardResult(
            backtest=result,
            targets=targets,
            folds=folds,
            fold_metrics=pd.DataFrame(rows).set_index("fold"),
            oos_signals=oos_signals,
            oos_start=oos_start,
        )
