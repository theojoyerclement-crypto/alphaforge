"""Engine accounting checked against hand calculations."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from alphaforge.backtest import (
    BacktestEngine,
    LinearCostModel,
    SquareRootImpactModel,
    make_walk_forward_folds,
)
from alphaforge.data.market_data import MarketData
from alphaforge.portfolio import DrawdownController


def _weights(data: MarketData, a: float, b: float) -> pd.DataFrame:
    return pd.DataFrame({"A": a, "B": b}, index=data.index)


def test_buy_and_hold_matches_asset_returns(constant_market: MarketData) -> None:
    engine = BacktestEngine(LinearCostModel(0, 0), execution_delay=0)
    res = engine.run(_weights(constant_market, 1.0, 0.0), constant_market)
    assert res.returns.iloc[0] == 0.0  # bought at the first close
    assert np.allclose(res.returns.iloc[1:], 0.01)
    assert res.equity.iloc[-1] == pytest.approx(1e6 * 1.01**9)


def test_execution_delay_shifts_pnl(constant_market: MarketData) -> None:
    engine = BacktestEngine(LinearCostModel(0, 0), execution_delay=2)
    res = engine.run(_weights(constant_market, 1.0, 0.0), constant_market)
    assert np.allclose(res.returns.iloc[:3], 0.0)
    assert np.allclose(res.returns.iloc[3:], 0.01)


def test_costs_charged_on_drifted_turnover(constant_market: MarketData) -> None:
    fee = 10.0
    engine = BacktestEngine(LinearCostModel(fee, 0), execution_delay=0)
    res = engine.run(_weights(constant_market, 0.5, 0.5), constant_market)
    assert res.turnover.iloc[0] == pytest.approx(1.0)
    assert res.costs.iloc[0] == pytest.approx(1.0 * fee * 1e-4)
    # after bar 1 the weights have drifted: rebalancing back to 50/50 costs a little
    assert 0 < res.turnover.iloc[1] < 0.05
    identity = res.gross_returns + res.funding - res.costs
    assert np.allclose(res.returns, identity, atol=1e-12)


def test_funding_paid_by_longs(constant_market: MarketData) -> None:
    funding = pd.DataFrame(0.001, index=constant_market.index, columns=constant_market.symbols)
    data = MarketData(
        constant_market.open,
        constant_market.high,
        constant_market.low,
        constant_market.close,
        constant_market.volume,
        "1d",
        funding=funding,
    )
    engine = BacktestEngine(LinearCostModel(0, 0), execution_delay=0)
    long = engine.run(_weights(data, 1.0, 0.0), data)
    short = engine.run(_weights(data, -1.0, 0.0), data)
    assert np.allclose(long.funding.iloc[1:], -0.001)
    assert np.allclose(short.funding.iloc[1:], 0.001)
    off = BacktestEngine(LinearCostModel(0, 0), execution_delay=0, apply_funding=False).run(
        _weights(data, 1.0, 0.0), data
    )
    assert (off.funding == 0).all()


def test_sqrt_impact_grows_with_size() -> None:
    model = SquareRootImpactModel(0, 0, impact_coefficient=1.0)
    vol, adv = np.array([0.02]), np.array([1e6])
    small = model.compute(np.array([0.01]), 1e6, vol, adv)[0] / 0.01
    big = model.compute(np.array([0.5]), 1e6, vol, adv)[0] / 0.5
    assert big / small == pytest.approx(np.sqrt(50), rel=1e-9)
    unknown = model.compute(np.array([0.01]), 1e6, vol, np.array([np.nan]))[0]
    assert unknown == pytest.approx(0.01 * 0.02)  # 100% participation fallback


def test_drawdown_controller_reduces_drawdown(market: MarketData) -> None:
    w = pd.DataFrame(1.0 / 8, index=market.index, columns=market.symbols).where(
        market.close.notna(), 0.0
    )
    free = BacktestEngine(execution_delay=1).run(w, market)
    controlled = BacktestEngine(
        execution_delay=1, drawdown_controller=DrawdownController(0.10, 0.05, cooldown_bars=10)
    ).run(w, market)
    assert (controlled.risk_scale == 0).any()
    assert (controlled.risk_scale < 1).mean() > 0.05
    assert controlled.metrics().max_drawdown > free.metrics().max_drawdown


def test_start_trims_warmup(market: MarketData) -> None:
    w = pd.DataFrame(0.1, index=market.index, columns=market.symbols)
    res = BacktestEngine().run(w, market, start="2021-06-01")
    assert res.returns.index[0] == pd.Timestamp("2021-06-01", tz="UTC")


def test_walk_forward_folds_tile_without_overlap() -> None:
    folds = make_walk_forward_folds(1000, train_bars=400, test_bars=150, embargo_bars=10)
    assert folds[0].train_end + 10 == folds[0].test_start
    for prev, nxt in itertools.pairwise(folds):
        assert prev.test_end == nxt.test_start
        assert nxt.train_end <= nxt.test_start - 10
    assert folds[-1].test_end == 1000
    rolling = make_walk_forward_folds(1000, 400, 150, expanding=False)
    assert all(f.train_end - f.train_start == 400 for f in rolling)
    with pytest.raises(ValueError):
        make_walk_forward_folds(100, 100, 10)
