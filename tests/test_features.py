"""Causality is the single most important property of a feature: test it for all of them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.data.market_data import MarketData
from alphaforge.features import (
    FeatureEngine,
    compute_feature,
    cs_rank,
    list_features,
    long_short_quantiles,
)
from tests.conftest import perturb_future

CUTOFF = 500


@pytest.mark.parametrize("name", [f.name for f in list_features()])
def test_feature_has_no_lookahead(market: MarketData, name: str) -> None:
    original = compute_feature(name, market)
    perturbed = compute_feature(name, perturb_future(market, CUTOFF))
    pd.testing.assert_frame_equal(
        original.iloc[:CUTOFF], perturbed.iloc[:CUTOFF], check_exact=False, rtol=1e-9
    )


@pytest.mark.parametrize("name", [f.name for f in list_features()])
def test_feature_shape_and_masking(market: MarketData, name: str) -> None:
    values = compute_feature(name, market)
    assert values.shape == market.close.shape
    assert values.where(market.close.isna()).isna().all().all(), "untradable bars must be NaN"
    assert not np.isinf(values.to_numpy(dtype=float)).any()


def test_feature_values_are_sensible(market: MarketData) -> None:
    rsi = compute_feature("rsi", market).stack().dropna()
    assert rsi.between(0, 100).all()
    er = compute_feature("efficiency_ratio", market).stack().dropna()
    assert er.between(0, 1 + 1e-12).all()
    vol = compute_feature("realized_vol", market).stack().dropna()
    assert 0.2 < vol.median() < 2.0  # annualised crypto-like vol


def test_engine_long_matrix(market: MarketData) -> None:
    engine = FeatureEngine(["rsi", ("momentum", {"lookback": 10}), "funding_rate"])
    X = engine.to_long(market, cs_normalize=True)
    assert list(X.columns) == ["rsi", "momentum_lookback10", "funding_rate"]
    assert X.index.names == ["timestamp", "symbol"]
    assert X.dropna().abs().max().max() <= 0.5


def test_cs_rank_and_quantile_portfolio() -> None:
    score = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0, np.nan]], columns=list("abcdef"))
    ranks = cs_rank(score)
    assert ranks.iloc[0, 4] == pytest.approx(0.5)
    w = long_short_quantiles(score, quantile=0.4)
    assert w.iloc[0].sum() == pytest.approx(0.0)
    assert w.iloc[0].abs().sum() == pytest.approx(1.0)
    assert w.iloc[0]["e"] > 0 and w.iloc[0]["a"] < 0 and w.iloc[0]["f"] == 0
