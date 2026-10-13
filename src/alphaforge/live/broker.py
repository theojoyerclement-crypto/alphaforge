"""Execution adapters: the only layer that differs between a backtest and a live book.

:class:`Broker` is the contract a venue adapter must satisfy. :class:`PaperBroker` is the
reference implementation: it fills at the bar close and reuses the backtest engine's own
mark-to-market function, cost model and cost inputs, so a paper account and a backtest fed
the same targets produce the same equity (asserted in ``tests/test_live.py``). A real adapter
(CCXT orders, an exchange's REST API) would implement the same three methods by reading
balances and sending orders; nothing upstream of it changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.backtest.engine import BacktestEngine, mark_to_market
from alphaforge.data.market_data import MarketData
from alphaforge.utils.logging import get_logger
from alphaforge.utils.time import to_utc

logger = get_logger(__name__)


@dataclass
class AccountState:
    """Serializable account snapshot (NAV weights, not units, as in the backtest)."""

    nav: float
    weights: dict[str, float] = field(default_factory=dict)
    last_bar: str | None = None  # ISO timestamp of the last bar the account was marked to

    def to_dict(self) -> dict[str, Any]:
        return {"nav": self.nav, "weights": dict(self.weights), "last_bar": self.last_bar}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AccountState:
        return cls(
            nav=float(raw["nav"]),
            weights={str(k): float(v) for k, v in raw.get("weights", {}).items()},
            last_bar=raw.get("last_bar"),
        )


class Broker(ABC):
    """What the trading loop needs from a venue."""

    @property
    @abstractmethod
    def nav(self) -> float:
        """Current account value."""

    @property
    @abstractmethod
    def weights(self) -> pd.Series:
        """Current positions as fractions of NAV."""

    @property
    @abstractmethod
    def last_bar(self) -> pd.Timestamp | None:
        """Last bar the account was brought up to (``None`` for a fresh account)."""

    @abstractmethod
    def sync(self, data: MarketData) -> pd.DataFrame:
        """Bring the account up to the last bar of ``data``; return one row per bar applied."""

    @abstractmethod
    def execute(self, target: pd.Series, data: MarketData) -> pd.DataFrame:
        """Trade to ``target`` NAV weights at the last bar of ``data``; return the fills."""


class PaperBroker(Broker):
    """Simulated account that fills at the close with the backtest's cost model.

    Args:
        engine: Supplies the cost model, its volatility/ADV estimators and the funding
            convention, i.e. the exact frictions the backtest charged.
        state: Snapshot to resume from; a fresh account starts at
            ``engine.initial_capital`` with no positions.
    """

    def __init__(self, engine: BacktestEngine, state: AccountState | None = None) -> None:
        self.engine = engine
        self.state = state or AccountState(nav=engine.initial_capital)

    @property
    def nav(self) -> float:
        return self.state.nav

    @property
    def weights(self) -> pd.Series:
        return pd.Series(self.state.weights, dtype=float)

    @property
    def last_bar(self) -> pd.Timestamp | None:
        return None if self.state.last_bar is None else to_utc(self.state.last_bar)

    def _book(self, symbols: pd.Index) -> np.ndarray:
        held = self.weights
        unknown = held.index.difference(symbols)
        if (held[unknown].abs() > 0).any():
            raise ValueError(f"Account holds {list(unknown)} which are missing from the data")
        return held.reindex(symbols).fillna(0.0).to_numpy()

    def sync(self, data: MarketData) -> pd.DataFrame:
        """Mark positions to every bar after ``last_bar``; positions drift, nothing trades.

        Bars missed between two runs are applied one by one (a book cannot rebalance at a
        bar it did not see), which keeps the compounding identical to the engine's.
        """
        symbols = data.close.columns
        index = data.index
        if self.state.last_bar is None:
            self.state.last_bar = index[-1].isoformat()
            return pd.DataFrame(columns=["nav", "price_pnl", "funding_pnl"])
        last = to_utc(self.state.last_bar)
        new_bars = index[index > last]
        if len(new_bars) == 0:
            return pd.DataFrame(columns=["nav", "price_pnl", "funding_pnl"])
        if last not in index:
            raise ValueError(f"Last marked bar {last} is not in the data; cannot mark to market")

        returns = data.returns().fillna(0.0).loc[new_bars]
        if self.engine.apply_funding and data.funding is not None:
            funding = data.funding.reindex(index=new_bars, columns=symbols).fillna(0.0)
        else:
            funding = pd.DataFrame(0.0, index=new_bars, columns=symbols)

        w = self._book(symbols)
        nav = self.state.nav
        rows = []
        for bar in new_bars:
            price_pnl, funding_pnl, growth, w = mark_to_market(
                w, returns.loc[bar].to_numpy(), funding.loc[bar].to_numpy()
            )
            nav = nav * growth if growth > 0 else 0.0
            rows.append(
                {"bar": bar, "nav": nav, "price_pnl": price_pnl, "funding_pnl": funding_pnl}
            )
        self.state.nav = nav
        self.state.weights = {s: float(x) for s, x in zip(symbols, w, strict=True) if x != 0.0}
        self.state.last_bar = new_bars[-1].isoformat()
        return pd.DataFrame(rows).set_index("bar")

    def execute(self, target: pd.Series, data: MarketData) -> pd.DataFrame:
        """Fill ``target`` at the last close; untradable (NaN close) assets are forced flat."""
        symbols = data.close.columns
        bar = data.index[-1]
        if self.state.last_bar is None or to_utc(self.state.last_bar) != bar:
            raise ValueError("sync() the account to the latest bar before executing")
        if self.state.nav <= 0:
            logger.error("Paper account is wiped out; not trading")
            return pd.DataFrame()

        close = data.close.iloc[-1]
        desired = target.reindex(symbols).fillna(0.0).where(close.notna(), 0.0).to_numpy()
        before = self._book(symbols)
        vol_frame, adv_frame = self.engine.cost_inputs(data)
        vol, adv = vol_frame.iloc[-1].to_numpy(), adv_frame.iloc[-1].to_numpy()
        nav_pre = self.state.nav
        # NAV moves through the engine's own trade step; per-asset costs are only for the log.
        self.state.nav, _, trades = self.engine.trade(before, nav_pre, desired, vol, adv)
        cost = self.engine.cost_model.compute(trades, nav_pre, vol, adv)
        self.state.weights = {
            s: float(x) for s, x in zip(symbols, desired, strict=True) if x != 0.0
        }

        fills = pd.DataFrame(
            {
                "bar": bar,
                "symbol": symbols,
                "price": close.to_numpy(),
                "weight_before": before,
                "weight_after": desired,
                "trade_weight": trades,
                "trade_notional": trades * nav_pre,
                "cost": cost * nav_pre,
            }
        )
        return fills[fills["trade_weight"].abs() > 1e-12].reset_index(drop=True)
