from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.data.market_data import MarketData
from alphaforge.strategies import available_strategies, build_strategy
from alphaforge.strategies.mean_reversion import hysteresis_positions
from tests.conftest import perturb_future

RULE_BASED = sorted(n for n, c in available_strategies().items() if not c.requires_fit)
FITTED = sorted(n for n, c in available_strategies().items() if c.requires_fit)


@pytest.mark.parametrize("name", RULE_BASED)
def test_signals_bounded_and_aligned(market: MarketData, name: str) -> None:
    signals = build_strategy(name).generate_signals(market)
    assert signals.shape == market.close.shape
    assert signals.notna().all().all()
    assert signals.abs().max().max() <= 1.0 + 1e-12
    assert (signals.where(market.close.isna()).fillna(0.0) == 0.0).all().all()
    assert signals.abs().sum().sum() > 0, "strategy never trades"


@pytest.mark.parametrize("name", RULE_BASED)
def test_rule_based_strategies_are_causal(market: MarketData, name: str) -> None:
    cutoff = 550
    a = build_strategy(name).generate_signals(market)
    b = build_strategy(name).generate_signals(perturb_future(market, cutoff))
    pd.testing.assert_frame_equal(a.iloc[:cutoff], b.iloc[:cutoff])


@pytest.mark.parametrize("name", FITTED)
def test_fitted_strategies_train_and_predict(market: MarketData, name: str) -> None:
    params = {"model": "logistic"} if name == "ml_directional" else {}
    strategy = build_strategy(name, params)
    with pytest.raises(RuntimeError):
        strategy.generate_signals(market)
    strategy.fit(market.iloc(0, 500))
    signals = strategy.generate_signals(market)
    assert signals.shape == market.close.shape
    assert signals.abs().max().max() <= 1.0 + 1e-12


def test_fitted_strategy_out_of_sample_is_causal(market: MarketData) -> None:
    strategy = build_strategy("ml_directional", {"model": "logistic"}).fit(market.iloc(0, 500))
    a = strategy.generate_signals(market)
    b = strategy.generate_signals(perturb_future(market, 650))
    pd.testing.assert_frame_equal(a.iloc[:650], b.iloc[:650])


def test_hysteresis_holds_between_bands() -> None:
    z = np.array([[0.0], [2.0], [1.0], [0.5], [0.1], [-2.0], [-0.1]])
    pos = hysteresis_positions(z, entry=1.5, exit_=0.25).ravel()
    assert pos.tolist() == [0, -1, -1, -1, 0, 1, 0]


def test_cross_sectional_strategies_are_dollar_neutral(market: MarketData) -> None:
    for name in ("xs_momentum", "funding_carry", "st_reversal"):
        signals = build_strategy(name).generate_signals(market)
        active = signals.abs().sum(axis=1) > 0
        assert signals[active].sum(axis=1).abs().max() < 1e-9, name


def test_unknown_strategy_raises() -> None:
    with pytest.raises(KeyError):
        build_strategy("does_not_exist")
