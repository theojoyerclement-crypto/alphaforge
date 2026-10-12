"""Model factory and the pooled cross-asset directional model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from alphaforge.backtest.cv import PurgedKFold
from alphaforge.data.market_data import MarketData
from alphaforge.features.base import FeatureEngine, FeatureRequest
from alphaforge.features.transforms import panel_to_long
from alphaforge.ml.labeling import fixed_horizon_labels
from alphaforge.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_ML_FEATURES: list[tuple[str, dict[str, Any]]] = [
    ("returns", {"lookback": 1}),
    ("returns", {"lookback": 5}),
    ("momentum", {"lookback": 20, "skip": 1}),
    ("momentum", {"lookback": 60, "skip": 1}),
    ("vol_adj_momentum", {"lookback": 60}),
    ("realized_vol", {"lookback": 20}),
    ("parkinson_vol", {"lookback": 20}),
    ("zscore", {"lookback": 20}),
    ("rsi", {"lookback": 14}),
    ("efficiency_ratio", {"lookback": 20}),
    ("volume_zscore", {"lookback": 20}),
    ("amihud", {"lookback": 20}),
    ("close_location", {"lookback": 10}),
    ("funding_rate", {"lookback": 21}),
    ("funding_zscore", {"lookback": 90}),
]


def make_classifier(name: str, random_state: int = 42, **params: Any) -> ClassifierMixin | Pipeline:
    """Build a probabilistic classifier by name.

    ``"lightgbm"`` falls back to scikit-learn's histogram gradient boosting when LightGBM is
    not installed, so the framework runs on a minimal environment.
    """
    if name == "logistic":
        return make_pipeline(
            StandardScaler(), LogisticRegression(C=params.pop("C", 0.1), max_iter=1000, **params)
        )
    if name == "random_forest":
        defaults: dict[str, Any] = {
            "n_estimators": 300,
            "max_depth": 5,
            "min_samples_leaf": 50,
            "n_jobs": -1,
        }
        return RandomForestClassifier(random_state=random_state, **{**defaults, **params})
    if name == "hist_gb":
        defaults = {
            "max_depth": 3,
            "learning_rate": 0.05,
            "max_iter": 200,
            "min_samples_leaf": 100,
        }
        return HistGradientBoostingClassifier(random_state=random_state, **{**defaults, **params})
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            logger.warning("lightgbm not installed; using HistGradientBoostingClassifier")
            return make_classifier("hist_gb", random_state, **params)
        defaults = {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 100,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "verbose": -1,
        }
        return LGBMClassifier(random_state=random_state, **{**defaults, **params})
    raise ValueError(f"Unknown model {name!r}; use logistic, random_forest, hist_gb or lightgbm")


def label_end_times(index: pd.MultiIndex, bar_index: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """``t1`` for pooled samples: the timestamp ``horizon`` bars after each sample's bar."""
    times = index.get_level_values("timestamp")
    pos = bar_index.get_indexer(times)
    end = np.minimum(pos + horizon, len(bar_index) - 1)
    return pd.Series(bar_index[end], index=index, name="t1")


class DirectionalModel:
    """Pooled cross-asset classifier of the ``horizon``-bar forward return sign.

    Features are cross-sectionally ranked so one model serves all assets. The target is
    whether an asset will *outperform the cross-sectional median* (``relative=True``,
    market-neutral) or simply rise.

    Args:
        model: Name passed to :func:`make_classifier`.
        horizon: Prediction horizon in bars.
        features: Feature requests (defaults to :data:`DEFAULT_ML_FEATURES`).
        relative: Predict relative (cross-sectional) rather than absolute direction.
    """

    def __init__(
        self,
        model: str = "lightgbm",
        horizon: int = 5,
        features: Iterable[FeatureRequest | str | tuple[str, Mapping[str, Any]]] | None = None,
        relative: bool = True,
        random_state: int = 42,
        model_params: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = model
        self.horizon = horizon
        self.relative = relative
        self.random_state = random_state
        self.model_params = dict(model_params or {})
        self.engine = FeatureEngine(features if features is not None else DEFAULT_ML_FEATURES)
        self.feature_names_: list[str] | None = None
        self.estimator_: Any = None

    def build_dataset(self, data: MarketData) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Design matrix ``X``, binary target ``y`` and label end times ``t1``."""
        X = self.engine.to_long(data, cs_normalize=True)
        _, fwd = fixed_horizon_labels(data.close, self.horizon)
        if self.relative:
            fwd = fwd.sub(fwd.median(axis=1), axis=0)
        target = panel_to_long({"fwd": fwd})["fwd"]
        frame = X.join(target).dropna()
        y = (frame.pop("fwd") > 0).astype(int)
        t1 = label_end_times(frame.index, data.index, self.horizon)
        return frame, y, t1

    def fit(self, data: MarketData) -> DirectionalModel:
        X, y, _ = self.build_dataset(data)
        if len(X) < 200:
            raise ValueError(f"Not enough training samples ({len(X)})")
        self.feature_names_ = list(X.columns)
        self.estimator_ = make_classifier(self.model_name, self.random_state, **self.model_params)
        self.estimator_.fit(X.to_numpy(), y.to_numpy())
        logger.info("Fitted %s on %d samples x %d features", self.model_name, *X.shape)
        return self

    def predict_proba(self, data: MarketData) -> pd.DataFrame:
        """Wide panel of P(up) (or P(outperform)); ``NaN`` where features are incomplete."""
        if self.estimator_ is None or self.feature_names_ is None:
            raise RuntimeError("Model is not fitted")
        X = self.engine.to_long(data, cs_normalize=True).reindex(columns=self.feature_names_)
        valid = X.notna().all(axis=1)
        proba = pd.Series(np.nan, index=X.index)
        if valid.any():
            proba[valid] = self.estimator_.predict_proba(X[valid].to_numpy())[:, 1]
        return proba.unstack("symbol").reindex(index=data.index, columns=data.close.columns)

    def cross_validate(
        self, data: MarketData, n_splits: int = 5, embargo_pct: float = 0.02
    ) -> pd.DataFrame:
        """Purged K-fold scores (accuracy, ROC AUC, log loss) per fold."""
        X, y, t1 = self.build_dataset(data)
        t0 = X.index.get_level_values("timestamp")
        order = np.argsort(t0.asi8, kind="stable")
        X, y, t1, t0 = X.iloc[order], y.iloc[order], t1.iloc[order], t0[order]
        rows = []
        for fold, (train, test) in enumerate(PurgedKFold(n_splits, embargo_pct).split(t0, t1)):
            est = clone(make_classifier(self.model_name, self.random_state, **self.model_params))
            est.fit(X.iloc[train].to_numpy(), y.iloc[train].to_numpy())
            proba = est.predict_proba(X.iloc[test].to_numpy())[:, 1]
            y_test = y.iloc[test].to_numpy()
            rows.append(
                {
                    "fold": fold,
                    "n_train": len(train),
                    "n_test": len(test),
                    "accuracy": accuracy_score(y_test, proba > 0.5),
                    "auc": roc_auc_score(y_test, proba) if len(set(y_test)) > 1 else np.nan,
                    "log_loss": log_loss(y_test, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]),
                }
            )
        return pd.DataFrame(rows).set_index("fold")
