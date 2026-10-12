"""Meta-labeling: a secondary model decides *whether and how much* to act on a primary signal.

The primary strategy sets the side. A classifier, trained on triple-barrier outcomes of past
primary trades, estimates P(trade is profitable). Bet size is a monotone function of that
probability, so low-confidence trades are skipped. This typically raises precision and
Sharpe at the cost of fewer trades (Lopez de Prado, 2018, ch. 3).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from alphaforge.data.market_data import MarketData
from alphaforge.features.base import FeatureEngine, FeatureRequest
from alphaforge.features.transforms import panel_to_long
from alphaforge.ml.labeling import ewma_volatility, triple_barrier_labels
from alphaforge.ml.models import DEFAULT_ML_FEATURES, make_classifier
from alphaforge.utils.logging import get_logger

logger = get_logger(__name__)


def probability_to_size(proba: np.ndarray | pd.DataFrame, min_probability: float = 0.5) -> Any:
    """Map P(success) to a bet size in ``[0, 1]`` via ``2 * Phi(z) - 1`` with
    ``z = (p - 0.5) / sqrt(p (1 - p))``; zero below ``min_probability``."""
    p = np.clip(proba, 1e-6, 1 - 1e-6)
    z = (p - 0.5) / np.sqrt(p * (1 - p))
    size = 2.0 * stats.norm.cdf(z) - 1.0
    size = np.where(p >= min_probability, np.clip(size, 0.0, 1.0), 0.0)
    if isinstance(proba, pd.DataFrame):
        return pd.DataFrame(size, index=proba.index, columns=proba.columns).where(proba.notna())
    return size


class MetaLabeler:
    """Secondary classifier sizing a primary strategy's trades.

    Args:
        model: Classifier name (see :func:`make_classifier`).
        pt_mult, sl_mult, max_holding: Triple-barrier parameters.
        vol_halflife: Half-life of the EWMA vol that scales the barriers.
        min_probability: Trades with P(success) below this are skipped.
        features: Context features (defaults to :data:`DEFAULT_ML_FEATURES`).
    """

    def __init__(
        self,
        model: str = "random_forest",
        pt_mult: float = 1.5,
        sl_mult: float = 1.0,
        max_holding: int = 10,
        vol_halflife: int = 20,
        min_probability: float = 0.5,
        features: Iterable[FeatureRequest | str | tuple[str, Mapping[str, Any]]] | None = None,
        random_state: int = 42,
        model_params: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = model
        self.pt_mult = pt_mult
        self.sl_mult = sl_mult
        self.max_holding = max_holding
        self.vol_halflife = vol_halflife
        self.min_probability = min_probability
        self.random_state = random_state
        self.model_params = dict(model_params or {})
        self.engine = FeatureEngine(features if features is not None else DEFAULT_ML_FEATURES)
        self.feature_names_: list[str] | None = None
        self.estimator_: Any = None

    def _features(self, data: MarketData, primary: pd.DataFrame) -> pd.DataFrame:
        X = self.engine.to_long(data, cs_normalize=True)
        extra = panel_to_long({"side": np.sign(primary), "conviction": primary.abs()})
        return X.join(extra)

    def build_dataset(
        self, data: MarketData, primary: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Features, meta-labels (1 = primary trade profitable) and label end times."""
        vol = ewma_volatility(data.close, self.vol_halflife)
        frames = []
        for symbol in data.symbols:
            side = np.sign(primary[symbol])
            events = side.index[(side != 0) & side.notna()]
            if len(events) == 0:
                continue
            tb = triple_barrier_labels(
                data.close[symbol].dropna(),
                events.intersection(data.close[symbol].dropna().index),
                vol[symbol],
                self.pt_mult,
                self.sl_mult,
                self.max_holding,
                side=side,
            )
            tb["symbol"] = symbol
            frames.append(tb)
        if not frames:
            raise ValueError("Primary strategy produced no trades to meta-label")
        labels = pd.concat(frames).reset_index().set_index(["t0", "symbol"]).sort_index()
        labels.index = labels.index.set_names(["timestamp", "symbol"])
        X = self._features(data, primary).reindex(labels.index)
        valid = X.notna().all(axis=1)
        return X[valid], labels.loc[valid, "label"].astype(int), labels.loc[valid, "t1"]

    def fit(self, data: MarketData, primary: pd.DataFrame) -> MetaLabeler:
        X, y, _ = self.build_dataset(data, primary)
        if y.nunique() < 2:
            raise ValueError("Meta-labels contain a single class; cannot fit")
        self.feature_names_ = list(X.columns)
        self.estimator_ = make_classifier(self.model_name, self.random_state, **self.model_params)
        self.estimator_.fit(X.to_numpy(), y.to_numpy())
        logger.info("Meta-labeler fitted on %d trades (hit rate %.1f%%)", len(y), 100 * y.mean())
        return self

    def predict_proba(self, data: MarketData, primary: pd.DataFrame) -> pd.DataFrame:
        """P(primary trade succeeds) per bar and asset (``NaN`` where there is no trade)."""
        if self.estimator_ is None or self.feature_names_ is None:
            raise RuntimeError("MetaLabeler is not fitted")
        X = self._features(data, primary).reindex(columns=self.feature_names_)
        active = X["side"].fillna(0) != 0
        valid = X.notna().all(axis=1) & active
        proba = pd.Series(np.nan, index=X.index)
        if valid.any():
            proba[valid] = self.estimator_.predict_proba(X[valid].to_numpy())[:, 1]
        return proba.unstack("symbol").reindex(index=data.index, columns=data.close.columns)

    def size_positions(self, data: MarketData, primary: pd.DataFrame) -> pd.DataFrame:
        """Primary side x meta bet size."""
        size = probability_to_size(self.predict_proba(data, primary), self.min_probability)
        return (np.sign(primary) * size).fillna(0.0)
