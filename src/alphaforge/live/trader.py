"""Paper-trading loop that runs the backtest's decision code, one bar at a time.

``PaperTrader.step(data)`` is what a scheduler would call after each bar closes:

1. ``broker.sync``: mark the account to the new bar(s).
2. Signals from the configured strategies and targets from :class:`PortfolioConstructor`,
   built from data up to the current close only.
3. Drawdown overlay, as in the backtest: with ``drawdown_basis: shadow`` the trader keeps a
   second paper account that always holds the full-exposure target, and the controller
   watches *its* NAV; with ``realized`` it watches the real account.
4. ``broker.execute`` the target decided ``execution_delay`` bars ago, times the multiplier.

Step 4 takes row ``-1 - delay`` of a target panel computed *now*. That equals what was
decided at that bar only because every strategy and the portfolio constructor are causal;
:meth:`PaperTrader.replay` checks it end to end by reproducing the backtest's equity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.backtest.engine import BacktestEngine
from alphaforge.config.schema import ExperimentConfig
from alphaforge.data.market_data import MarketData
from alphaforge.live.broker import AccountState, Broker, PaperBroker
from alphaforge.portfolio.construction import PortfolioConstructor
from alphaforge.research.experiment import ConfigurationError
from alphaforge.strategies import Strategy, build_strategy
from alphaforge.utils.logging import get_logger
from alphaforge.utils.time import to_utc

logger = get_logger(__name__)


@dataclass
class StepReport:
    """Outcome of one :meth:`PaperTrader.step`."""

    bar: pd.Timestamp
    nav_before: float  # after marking to market, before trading
    nav: float  # after trading costs
    risk_scale: float
    target: pd.Series
    fills: pd.DataFrame
    skipped: bool = False  # the bar had already been processed
    shadow_nav: float | None = None  # full-exposure account the overlay watches (v2)


class PaperTrader:
    """Drive a :class:`Broker` with the same strategies and portfolio code as the backtest.

    Args:
        config: The experiment config the backtest used. Strategies that learn
            (``requires_fit``) are refused: paper trading them needs persisted, versioned
            models, which is out of scope for this minimal loop.
        broker: Execution adapter; defaults to a fresh :class:`PaperBroker`.
    """

    def __init__(self, config: ExperimentConfig, broker: Broker | None = None) -> None:
        self.config = config
        self.strategies: dict[str, Strategy] = {
            spec.label: build_strategy(spec.name, spec.params) for spec in config.strategies
        }
        needs_fit = [label for label, s in self.strategies.items() if s.requires_fit]
        if needs_fit:
            raise ConfigurationError(
                f"Strategies {needs_fit} must be fitted; paper trading supports rule-based "
                "strategies only (persisted models are on the roadmap)"
            )
        self.constructor = PortfolioConstructor(config.portfolio, config.data.universe)
        self.engine = BacktestEngine.from_config(config.backtest, config.costs, config.risk)
        self.controller = self.engine.drawdown_controller
        self.delay = config.backtest.execution_delay
        self.broker: Broker = broker or PaperBroker(self.engine)
        use_shadow = self.controller is not None and self.engine.drawdown_basis == "shadow"
        self.shadow: PaperBroker | None = PaperBroker(self.engine) if use_shadow else None

    def decide(self, data: MarketData) -> pd.Series:
        """Target weights to trade at the last bar: the row decided ``delay`` bars ago."""
        signals = {label: s.generate_signals(data) for label, s in self.strategies.items()}
        weights = self.constructor.build(signals, data).weights
        if len(weights) <= self.delay:
            return pd.Series(0.0, index=data.close.columns)
        return weights.iloc[-1 - self.delay]

    def step(self, data: MarketData) -> StepReport:
        """Process the latest bar of ``data`` (idempotent if it was already processed)."""
        bar = data.index[-1]
        broker = self.broker
        if broker.last_bar is not None and broker.last_bar >= bar:
            logger.info("Bar %s already processed; nothing to do", bar)
            empty = pd.Series(dtype=float)
            return StepReport(bar, broker.nav, broker.nav, 1.0, empty, pd.DataFrame(), True)

        broker.sync(data)
        nav_before = broker.nav
        full = self.decide(data)
        basis = nav_before
        if self.shadow is not None:
            self.shadow.sync(data)
            basis = self.shadow.nav
            self.shadow.execute(full, data)
        controller = self.controller
        if controller is None or nav_before <= 0:
            scale = 1.0
        else:
            scale = controller.update(basis) if basis > 0 else 0.0
        target = full * scale
        fills = broker.execute(target, data)
        shadow_nav = None if self.shadow is None else self.shadow.nav
        return StepReport(bar, nav_before, broker.nav, scale, target, fills, shadow_nav=shadow_nav)

    def replay(self, data: MarketData, start: str | pd.Timestamp) -> pd.DataFrame:
        """Step through every bar from ``start``, each time seeing only data up to that bar.

        Returns one row per bar (NAV before/after trading, risk scale, turnover). Fed the
        same config, this must match :meth:`BacktestEngine.run` from ``start``.
        """
        first = int(data.index.searchsorted(to_utc(start)))
        if first >= data.n_bars:
            raise ValueError("start is after the last bar")
        rows = []
        for stop in range(first + 1, data.n_bars + 1):
            report = self.step(data.iloc(0, stop))
            turnover = float(report.fills["trade_weight"].abs().sum()) if len(report.fills) else 0.0
            rows.append(
                {
                    "bar": report.bar,
                    "nav_before": report.nav_before,
                    "nav": report.nav,
                    "risk_scale": report.risk_scale,
                    "turnover": turnover,
                }
            )
        return pd.DataFrame(rows).set_index("bar")

    # -- persistence -------------------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        """Write the account and risk-overlay state so the next run resumes from it."""
        if not isinstance(self.broker, PaperBroker):
            raise TypeError("Only PaperBroker state is persisted locally")
        payload: dict[str, Any] = {
            "experiment": self.config.name,
            "account": self.broker.state.to_dict(),
            "shadow_account": None if self.shadow is None else self.shadow.state.to_dict(),
            "drawdown_controller": self.controller.state_dict() if self.controller else None,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_state(cls, config: ExperimentConfig, path: str | Path) -> PaperTrader:
        """Resume a paper session; a missing file starts a fresh account."""
        trader = cls(config)
        source = Path(path)
        if not source.exists():
            return trader
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("experiment") != config.name:
            raise ValueError(
                f"{source} belongs to experiment {payload.get('experiment')!r}, "
                f"not {config.name!r}; use another --state-dir"
            )
        trader.broker = PaperBroker(trader.engine, AccountState.from_dict(payload["account"]))
        if trader.shadow is not None:
            shadow_state = payload.get("shadow_account")
            if shadow_state is None:
                raise ValueError(
                    f"{source} has no shadow account (written with drawdown_basis: realized); "
                    "resume with that setting or start a new --state-dir"
                )
            trader.shadow = PaperBroker(trader.engine, AccountState.from_dict(shadow_state))
        if trader.controller is not None and payload.get("drawdown_controller"):
            trader.controller.load_state_dict(payload["drawdown_controller"])
        return trader
