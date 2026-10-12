from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.data.market_data import MarketData
from alphaforge.ml import (
    DirectionalModel,
    GaussianMixtureRegimeModel,
    MetaLabeler,
    VolatilityRegimeClassifier,
    fixed_horizon_labels,
    market_regime_features,
    probability_to_size,
    triple_barrier_labels,
)
from alphaforge.strategies import build_strategy


def _path(values: list[float]) -> pd.Series:
    idx = pd.date_range("2023-01-01", periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=idx)


def test_triple_barrier_hits_each_barrier() -> None:
    close = _path([100, 101, 103, 99, 98, 97, 96, 95, 94, 93, 92, 91])
    vol = pd.Series(0.02, index=close.index)
    tb = triple_barrier_labels(close, close.index[:2], vol, pt_mult=1.0, sl_mult=1.0, max_holding=3)
    assert tb.loc[close.index[0], "barrier"] == "profit"  # +3% >= +2% at bar 2
    assert tb.loc[close.index[0], "t1"] == close.index[2]
    assert tb.loc[close.index[1], "barrier"] == "stop"  # 103 then 99: -1.98%... then 98 <= -2%
    short = triple_barrier_labels(
        close, close.index[3:4], vol, 1.0, 1.0, 3, side=pd.Series(-1.0, index=close.index)
    )
    assert short.iloc[0]["label"] == 1.0  # short in a falling market is a winning meta-label


def test_triple_barrier_drops_incomplete_events() -> None:
    close = _path([100.0, 100.1, 100.2, 100.1])
    vol = pd.Series(0.5, index=close.index)
    tb = triple_barrier_labels(close, close.index, vol, 1, 1, max_holding=5)
    assert tb.empty


def test_fixed_horizon_labels() -> None:
    close = pd.DataFrame({"a": [100.0, 110, 99, 99]})
    labels, _ = fixed_horizon_labels(close, 1, threshold=0.005)
    assert labels["a"].tolist()[:3] == [1.0, -1.0, 0.0]
    assert np.isnan(labels["a"].iloc[-1])


def test_bet_size_is_monotone() -> None:
    p = np.array([0.4, 0.5, 0.55, 0.7, 0.9])
    size = probability_to_size(p, min_probability=0.5)
    assert size[0] == 0.0 and size[1] == 0.0
    assert np.all(np.diff(size[1:]) > 0)


def test_directional_model_purged_cv(market: MarketData) -> None:
    model = DirectionalModel("logistic", horizon=3)
    scores = model.cross_validate(market, n_splits=3)
    assert len(scores) == 3
    assert scores["auc"].between(0.3, 0.8).all()
    assert (scores["n_train"] > 0).all()


def test_meta_labeler_fit_and_size(market: MarketData) -> None:
    primary = build_strategy("ts_momentum").generate_signals(market)
    train = market.iloc(0, 500)
    meta = MetaLabeler("logistic", max_holding=5).fit(train, primary.iloc[:500])
    sized = meta.size_positions(market, primary)
    assert sized.shape == primary.shape
    # meta-labeling may only shrink or skip trades, never flip the side
    assert ((np.sign(sized) == np.sign(primary)) | (sized == 0)).all().all()
    assert (sized.abs() <= primary.abs().clip(upper=1) + 1).all().all()


def test_regime_models(market: MarketData) -> None:
    feats = market_regime_features(market)
    gmm = GaussianMixtureRegimeModel(n_regimes=2).fit(feats.iloc[:500])
    labels = gmm.predict(feats)
    calm = feats["volatility"][labels == 0].mean()
    wild = feats["volatility"][labels == 1].mean()
    assert calm < wild  # regimes are ordered by volatility
    proba = gmm.predict_proba(feats).dropna()
    assert np.allclose(proba.sum(axis=1), 1.0)
    vol_regime = VolatilityRegimeClassifier(window=200).predict(feats["volatility"])
    assert set(vol_regime.dropna().unique()) <= {0.0, 1.0, 2.0}


def test_gmm_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        GaussianMixtureRegimeModel().predict(pd.DataFrame({"volatility": [0.1]}))
