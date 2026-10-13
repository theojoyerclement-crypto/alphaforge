"""Acceptance tests of drawdown overlay v2, pre-registered in docs/METHODOLOGY.md section 6.

They were written, and had to pass, before any reference config was re-run with the new rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.backtest import BacktestEngine, LinearCostModel
from alphaforge.config import ExperimentConfig
from alphaforge.data.market_data import MarketData
from alphaforge.live import PaperTrader
from alphaforge.portfolio import DrawdownController, PortfolioConstructor
from alphaforge.strategies import build_strategy
from tests.conftest import perturb_future


def _single_asset(path: np.ndarray) -> MarketData:
    index = pd.date_range("2022-01-01", periods=len(path), freq="1D", tz="UTC", name="timestamp")
    frame = pd.DataFrame(
        {"open": path, "high": path, "low": path, "close": path, "volume": 1e12 / path},
        index=index,
    )
    return MarketData.from_frames({"X": frame}, "1d")


def _engine(basis: str, soft: float = 0.15, hard: float = 0.30) -> BacktestEngine:
    return BacktestEngine(
        cost_model=LinearCostModel(0.0, 0.0),
        execution_delay=0,
        apply_funding=False,
        drawdown_controller=DrawdownController(hard, soft, cooldown_bars=20),
        drawdown_basis=basis,
    )


def _book(market: MarketData) -> pd.DataFrame:
    constructor = PortfolioConstructor()
    signals = {
        n: build_strategy(n).generate_signals(market) for n in ("ts_momentum", "xs_momentum")
    }
    return constructor.build(signals, market).weights


# 1. No trap -------------------------------------------------------------------------------


def test_exposure_returns_to_one_when_the_strategy_recovers() -> None:
    """A one-bar 28% gap (deep in the 15%-30% band), then a climb back just above the high.

    v1 sees its own de-risked equity: at 13% exposure the recovery barely moves it, so it is
    still trapped at the end although the strategy is back at its high.
    """
    gap = 100 * (1 - 0.28)
    data = _single_asset(np.concatenate([[100.0, 100.0, gap], np.linspace(gap, 101, 80)[1:]]))
    long_only = pd.DataFrame(1.0, index=data.index, columns=data.symbols)

    v2 = _engine("shadow").run(long_only, data)
    v1 = _engine("realized").run(long_only, data)
    assert v2.risk_scale.min() < 0.2, "the gap must push the overlay deep into the band"
    assert v2.risk_scale.iloc[-1] == 1.0, "v2 re-risks once the strategy is back at its high"
    # v1 is the documented trap: de-risked equity barely moves, so it never gets back to 1.
    assert v1.risk_scale.iloc[-1] < 1.0
    assert v2.equity.iloc[-1] > v1.equity.iloc[-1]


def test_hard_stop_cooldown_and_reset_on_shadow_basis() -> None:
    crash = 100 * (1 - 0.35 * np.linspace(0, 1, 15))
    flat = np.full(40, crash[-1])
    data = _single_asset(np.concatenate([crash, flat]))
    result = _engine("shadow").run(pd.DataFrame(1.0, index=data.index, columns=["X"]), data)
    zeros = np.flatnonzero(result.risk_scale.to_numpy() == 0.0)
    assert zeros.size >= 20, "flat for the cooldown after the 30% breach"
    assert result.risk_scale.iloc[-1] == 1.0, "after cooldown the shadow high-water mark resets"


# 2. Shadow equals overlay off -------------------------------------------------------------


@pytest.mark.parametrize("delay", [0, 1])
def test_shadow_equity_is_exactly_the_overlay_off_book(market: MarketData, delay: int) -> None:
    weights = _book(market)
    with_overlay = BacktestEngine(
        execution_delay=delay, drawdown_controller=DrawdownController(0.10, 0.03, 10)
    ).run(weights, market)
    off = BacktestEngine(execution_delay=delay).run(weights, market)
    assert with_overlay.shadow_equity is not None
    assert (with_overlay.risk_scale < 1).any(), "limits must bind for the test to mean anything"
    np.testing.assert_allclose(
        with_overlay.shadow_equity.to_numpy(), off.equity.to_numpy(), rtol=1e-12
    )
    assert "shadow_equity" in with_overlay.to_frame()


def test_realized_basis_reproduces_v1(market: MarketData) -> None:
    """v1 stays reproducible: the realized basis is the controller fed with the book's own NAV."""
    weights = _book(market)
    controller = DrawdownController(0.10, 0.03, 10)
    v1 = BacktestEngine(drawdown_controller=controller, drawdown_basis="realized").run(
        weights, market
    )
    assert v1.shadow_equity is None
    controller.reset()
    expected = [controller.update(nav) for nav in _pre_trade_navs(v1)]
    np.testing.assert_allclose(v1.risk_scale.to_numpy(), expected)


def _pre_trade_navs(result) -> list[float]:  # type: ignore[no-untyped-def]
    """NAV after marking to market and before trading, rebuilt from the accounting lines."""
    prev = np.concatenate([[1_000_000.0], result.equity.to_numpy()[:-1]])
    growth = 1.0 + result.gross_returns.to_numpy() + result.funding.to_numpy()
    return list(prev * growth)


# 3. Causality -----------------------------------------------------------------------------


@pytest.mark.parametrize("basis", ["shadow", "realized"])
def test_overlay_multipliers_are_causal(market: MarketData, basis: str) -> None:
    cutoff = 600
    weights = _book(market)
    engine = BacktestEngine(
        drawdown_controller=DrawdownController(0.10, 0.03, 10), drawdown_basis=basis
    )
    base = engine.run(weights, market).risk_scale
    scrambled = engine.run(weights, perturb_future(market, cutoff)).risk_scale
    pd.testing.assert_series_equal(base.iloc[:cutoff], scrambled.iloc[:cutoff])
    assert not base.iloc[cutoff:].equals(scrambled.iloc[cutoff:])


# 4. Paper parity --------------------------------------------------------------------------


@pytest.mark.parametrize("basis", ["shadow", "realized"])
def test_paper_replay_matches_backtest_under_both_bases(basis: str) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "overlay_parity",
            "data": {"symbols": ["A", "B", "C", "D", "E"], "synthetic": {"n_bars": 260}},
            "strategies": [{"name": "ts_momentum"}, {"name": "funding_carry"}],
            "portfolio": {"allocation_min_history": 20, "allocation_lookback": 40},
            "risk": {
                "max_drawdown": 0.06,
                "drawdown_soft_limit": 0.02,
                "cooldown_bars": 3,
                "drawdown_basis": basis,
            },
            "report": {"save": False},
        }
    )
    from alphaforge.data.loader import load_market_data

    data, _ = load_market_data(config.data, config.cleaning, config.seed)
    start = data.index[-60]
    trader = PaperTrader(config)
    paper = trader.replay(data, start)
    signals = {k: s.generate_signals(data) for k, s in trader.strategies.items()}
    weights = trader.constructor.build(signals, data).weights
    engine = BacktestEngine.from_config(config.backtest, config.costs, config.risk)
    backtest = engine.run(weights, data, start=start)

    assert (paper["risk_scale"] < 1).any()
    np.testing.assert_allclose(paper["nav"].to_numpy(), backtest.equity.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(
        paper["risk_scale"].to_numpy(), backtest.risk_scale.to_numpy(), atol=1e-12
    )
