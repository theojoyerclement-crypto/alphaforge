"""Feature registry and engine.

A *feature* is a pure function ``f(data: MarketData, **params) -> wide DataFrame`` whose row
``t`` depends only on data up to and including bar ``t``. Causality is enforced by a property
test (``tests/test_features.py``) that perturbs the future and checks the past is unchanged.

Register a new feature with::

    @register_feature("my_feature", category="technical", description="...")
    def my_feature(data: MarketData, lookback: int = 20) -> pd.DataFrame: ...
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.data.market_data import MarketData
from alphaforge.features.transforms import cs_rank, panel_to_long

FeatureFn = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class FeatureDefinition:
    """Metadata of a registered feature."""

    name: str
    func: FeatureFn
    category: str
    description: str
    defaults: Mapping[str, Any]


_REGISTRY: dict[str, FeatureDefinition] = {}


def register_feature(
    name: str, category: str, description: str = ""
) -> Callable[[FeatureFn], FeatureFn]:
    """Decorator registering a feature function under ``name``."""

    def decorator(func: FeatureFn) -> FeatureFn:
        if name in _REGISTRY:
            raise ValueError(f"Feature {name!r} already registered")
        params = list(inspect.signature(func).parameters.values())[1:]
        defaults = {p.name: p.default for p in params if p.default is not inspect.Parameter.empty}
        doc = description or (inspect.getdoc(func) or "").split("\n")[0]
        _REGISTRY[name] = FeatureDefinition(name, func, category, doc, defaults)
        return func

    return decorator


def get_feature(name: str) -> FeatureDefinition:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown feature {name!r}; available: {sorted(_REGISTRY)}") from exc


def list_features(category: str | None = None) -> list[FeatureDefinition]:
    """Registered features, optionally filtered by category."""
    return [f for f in _REGISTRY.values() if category is None or f.category == category]


def feature_label(name: str, params: Mapping[str, Any]) -> str:
    """Readable unique column name, e.g. ``momentum_lookback20_skip1``."""
    if not params:
        return name
    return name + "_" + "_".join(f"{k}{v}" for k, v in sorted(params.items()))


def compute_feature(name: str, data: MarketData, **params: Any) -> pd.DataFrame:
    """Compute a registered feature; values are masked where the asset is not tradable."""
    definition = get_feature(name)
    values = definition.func(data, **params)
    values = values.replace([np.inf, -np.inf], np.nan)
    return values.where(data.close.notna())


@dataclass(frozen=True)
class FeatureRequest:
    """A feature name plus parameter overrides."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return feature_label(self.name, self.params)

    @classmethod
    def parse(cls, item: FeatureRequest | str | tuple[str, Mapping[str, Any]]) -> FeatureRequest:
        if isinstance(item, FeatureRequest):
            return item
        if isinstance(item, str):
            return cls(item)
        name, params = item
        return cls(name, dict(params))


class FeatureEngine:
    """Computes a set of features and assembles model-ready matrices.

    Args:
        requests: Features to compute (names, ``(name, params)`` tuples or ``FeatureRequest``).
    """

    def __init__(
        self, requests: Iterable[FeatureRequest | str | tuple[str, Mapping[str, Any]]]
    ) -> None:
        self.requests: list[FeatureRequest] = [FeatureRequest.parse(r) for r in requests]
        labels = [r.label for r in self.requests]
        if len(labels) != len(set(labels)):
            raise ValueError("Duplicate feature requests")

    def compute(self, data: MarketData) -> dict[str, pd.DataFrame]:
        """Compute every requested feature as a wide panel, keyed by label."""
        return {r.label: compute_feature(r.name, data, **r.params) for r in self.requests}

    def to_long(
        self,
        data: MarketData,
        cs_normalize: bool = True,
        drop_empty: bool = True,
    ) -> pd.DataFrame:
        """Long ``(timestamp, symbol)`` design matrix.

        Args:
            cs_normalize: Replace each feature by its centred cross-sectional rank. This makes
                a model pooled across assets invariant to per-asset scale and drift.
            drop_empty: Drop features that are entirely missing (e.g. funding on spot data).
        """
        panels = self.compute(data)
        if drop_empty:
            panels = {k: v for k, v in panels.items() if v.notna().any().any()}
        if cs_normalize:
            panels = {k: cs_rank(v) for k, v in panels.items()}
        return panel_to_long(panels)

    @property
    def labels(self) -> Sequence[str]:
        return [r.label for r in self.requests]
