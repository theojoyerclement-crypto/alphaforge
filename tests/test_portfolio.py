from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from alphaforge.analytics.metrics import annualized_volatility
from alphaforge.config.schema import PortfolioConfig
from alphaforge.data.market_data import MarketData
from alphaforge.portfolio import (
    DrawdownController,
    PortfolioConstructor,
    PositionLimits,
    StrategyAllocator,
    apply_position_limits,
    kelly_fraction,
    risk_parity_weights,
    volatility_target,
)
from alphaforge.strategies import build_strategy


def test_volatility_target_hits_target(market: MarketData) -> None:
    weights = pd.DataFrame(1.0 / 8, index=market.index, columns=market.symbols).where(
        market.close.notna(), 0.0
    )
    scaled, _ = volatility_target(
        weights, market.returns(), 0.15, 20, market.periods_per_year, max_leverage=10
    )
    realised = (scaled.shift(1) * market.returns().fillna(0)).sum(axis=1).iloc[100:]
    assert annualized_volatility(realised, market.periods_per_year) == pytest.approx(0.15, rel=0.25)


def test_position_limits() -> None:
    w = pd.DataFrame([[0.9, 0.9, 0.9, -0.1]], columns=list("abcd"))
    out = apply_position_limits(w, PositionLimits(max_position=0.3, max_gross=0.8, max_net=0.5))
    assert out.abs().max().max() <= 0.3 + 1e-12
    assert out.abs().sum(axis=1).iloc[0] <= 0.8 + 1e-12
    assert abs(out.sum(axis=1).iloc[0]) <= 0.5 + 1e-12


def test_risk_parity_equalises_risk_contributions() -> None:
    cov = np.array([[0.04, 0.006, 0.0], [0.006, 0.01, 0.0], [0.0, 0.0, 0.0025]])
    w = risk_parity_weights(cov)
    rc = w * (cov @ w)
    assert w.sum() == pytest.approx(1.0)
    assert rc.std() / rc.mean() < 1e-10


@settings(max_examples=200, deadline=None)
@given(
    n=st.integers(2, 8),
    seed=st.integers(0, 2**32 - 1),
    collinearity=st.floats(-0.999, 0.999),
)
def test_risk_parity_property(n: int, seed: int, collinearity: float) -> None:
    """Any sample covariance, incl. (anti-)collinear sleeves: positive, sums to 1, equal RC."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(150, n)) * rng.uniform(1e-3, 0.1, n)
    base[:, 1] = collinearity * base[:, 0] + (1 - abs(collinearity)) * base[:, 1]
    cov = np.cov(base, rowvar=False)
    w = risk_parity_weights(cov)
    rc = w * (cov @ w)
    assert np.all(w > 0) and w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.abs(rc / rc.sum() - 1.0 / n).max() < 1e-9


def test_risk_parity_with_negatively_correlated_sleeves() -> None:
    """Regression: the former multiplicative fixed point returned ~[0.92, 0, 0.08] here, i.e. one
    sleeve carrying 97% of the risk, because a negative risk contribution blew up its update."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(120, 3))
    z[:, 1] = -0.9 * z[:, 0] + np.sqrt(1 - 0.81) * z[:, 1]
    cov = np.cov(z * [0.01, 0.02, 0.015], rowvar=False)
    w = risk_parity_weights(cov)
    rc = w * (cov @ w)
    assert np.all(w > 0.05)
    np.testing.assert_allclose(rc / rc.sum(), 1 / 3, atol=1e-10)


def test_risk_parity_ignores_dead_sleeves() -> None:
    cov = np.diag([0.01, 0.0, 0.04])
    w = risk_parity_weights(cov)
    assert w[1] == 0.0 and w[0] > w[2]


def test_allocator_is_causal_and_normalised() -> None:
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, [0.01, 0.02, 0.005], size=(400, 3)), columns=list("xyz"))
    alloc = StrategyAllocator(
        "risk_parity", lookback=100, rebalance_every=5, min_history=60
    ).allocate(rets)
    assert np.allclose(alloc.sum(axis=1), 1.0)
    assert alloc.iloc[200]["z"] > alloc.iloc[200]["y"]  # lower vol -> more capital
    shocked = rets.copy()
    shocked.iloc[300:] *= 10
    alloc2 = StrategyAllocator(
        "risk_parity", lookback=100, rebalance_every=5, min_history=60
    ).allocate(shocked)
    pd.testing.assert_frame_equal(alloc.iloc[:300], alloc2.iloc[:300])


def test_correlation_penalty_cuts_duplicate_sleeve() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0.001, 0.01, 300)
    rets = pd.DataFrame(
        {"a": base, "a_copy": base + rng.normal(0, 1e-4, 300), "b": rng.normal(0, 0.01, 300)}
    )
    with_pen = (
        StrategyAllocator("equal", correlation_threshold=0.9, min_history=60)
        .allocate(rets)
        .iloc[-1]
    )
    assert with_pen["b"] > with_pen["a"] or with_pen["b"] > with_pen["a_copy"]


def test_drawdown_controller_cycle() -> None:
    ctl = DrawdownController(max_drawdown=0.2, soft_limit=0.1, cooldown_bars=2)
    assert ctl.update(100) == 1.0
    assert ctl.update(85) == pytest.approx(0.5)  # 15% dd, halfway between limits
    assert ctl.update(79) == 0.0 and ctl.breaches == 1
    assert ctl.update(79) == 0.0  # cooldown
    assert ctl.update(79) == 0.0  # cooldown ends, HWM resets
    assert ctl.update(79) == 1.0


def test_kelly() -> None:
    assert kelly_fraction(0.1, 0.04, fraction=1.0) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        kelly_fraction(0.1, 0.0)


def test_constructor_end_to_end(market: MarketData) -> None:
    signals = {
        "trend": build_strategy("ts_momentum").generate_signals(market),
        "carry": build_strategy("funding_carry").generate_signals(market),
    }
    cfg = PortfolioConfig(max_position=0.25, max_gross_leverage=1.5)
    targets = PortfolioConstructor(cfg).build(signals, market)
    assert targets.weights.abs().max().max() <= 0.25 + 1e-9
    assert targets.weights.abs().sum(axis=1).max() <= 1.5 + 1e-9
    assert np.allclose(targets.allocation.sum(axis=1), 1.0)
    assert set(targets.sleeve_weights) == {"trend", "carry"}


@pytest.mark.parametrize("allocation", ["equal", "inverse_vol", "risk_parity"])
def test_portfolio_construction_is_causal(market: MarketData, allocation: str) -> None:
    """Targets at t must not move when the future is scrambled: the paper trader relies on it."""
    from tests.conftest import perturb_future

    cutoff = 600
    names = ("ts_momentum", "funding_carry", "zscore_reversion")
    constructor = PortfolioConstructor(PortfolioConfig(allocation=allocation))

    def targets(data: MarketData) -> pd.DataFrame:
        signals = {n: build_strategy(n).generate_signals(data) for n in names}
        return constructor.build(signals, data).weights

    base = targets(market)
    scrambled = targets(perturb_future(market, cutoff))
    pd.testing.assert_frame_equal(base.iloc[:cutoff], scrambled.iloc[:cutoff])
    assert not base.iloc[cutoff:].equals(scrambled.iloc[cutoff:])
