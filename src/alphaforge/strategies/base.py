"""Strategy contract and registry.

A strategy maps market data to *raw target positions* in ``[-1, 1]`` per asset and bar:

* row ``t`` may only use information available at the close of bar ``t``;
* the sign is the direction, the magnitude the conviction;
* sizing, risk targeting and costs are **not** the strategy's job; they belong to
  :mod:`alphaforge.portfolio` and :mod:`alphaforge.backtest`.

Strategies that learn parameters set ``requires_fit = True`` and implement :meth:`fit`. The
research runner refuses to evaluate them outside walk-forward, so a fitted model is never
scored on its own training data.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, TypeVar

import pandas as pd

from alphaforge.data.market_data import MarketData


class Strategy(ABC):
    """Base class of all strategies."""

    name: ClassVar[str] = "base"
    family: ClassVar[str] = "generic"
    requires_fit: ClassVar[bool] = False

    def fit(self, data: MarketData) -> Strategy:
        """Learn parameters from (training) data. No-op for rule-based strategies."""
        return self

    @abstractmethod
    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        """Raw target positions in ``[-1, 1]``, aligned with ``data.close``."""

    def describe(self) -> dict[str, Any]:
        """Name, family and public parameters (for reports and manifests)."""
        params = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return {"name": self.name, "family": self.family, "params": params}

    @staticmethod
    def finalize(signals: pd.DataFrame, data: MarketData) -> pd.DataFrame:
        """Clip to ``[-1, 1]``, align to the panel and zero untradable bars."""
        aligned = signals.reindex_like(data.close).clip(-1.0, 1.0)
        return aligned.where(data.close.notna(), 0.0).fillna(0.0)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.describe()["params"].items())
        return f"{type(self).__name__}({params})"


_S = TypeVar("_S", bound=type[Strategy])
_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(cls: _S) -> _S:
    """Class decorator adding a strategy to the registry under ``cls.name``."""
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"Strategy name {cls.name!r} already registered")
    _REGISTRY[cls.name] = cls
    return cls


def available_strategies() -> dict[str, type[Strategy]]:
    """Registered strategy classes by name."""
    return dict(_REGISTRY)


def build_strategy(name: str, params: Mapping[str, Any] | None = None) -> Strategy:
    """Instantiate a registered strategy."""
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown strategy {name!r}; available: {sorted(_REGISTRY)}") from exc
    kwargs = dict(params or {})
    signature = inspect.signature(cls)
    if not any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
        unknown = sorted(set(kwargs) - set(signature.parameters))
        if unknown:
            raise TypeError(
                f"Strategy {name!r} has no parameter(s) {unknown}; "
                f"accepted: {list(signature.parameters)}"
            )
    return cls(**kwargs)


StrategyFactory = Callable[[], dict[str, Strategy]]
