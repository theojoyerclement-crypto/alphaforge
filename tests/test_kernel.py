"""The compiled loop must reproduce the Python reference loop to 1e-10, in every situation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from alphaforge.backtest import BacktestEngine, LinearCostModel, SquareRootImpactModel, _kernel
from alphaforge.backtest.costs import CostModel
from alphaforge.data.market_data import MarketData
from alphaforge.portfolio import DrawdownController

pytestmark = pytest.mark.skipif(not _kernel.AVAILABLE, reason="numba not installed")

SERIES = (
    "returns",
    "gross_returns",
    "funding",
    "costs",
    "turnover",
    "equity",
    "risk_scale",
)


def _market(n_bars: int, n_assets: int, seed: int, listing_gaps: bool) -> MarketData:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n_bars, freq="1D", tz="UTC", name="timestamp")
    frames, funding = {}, {}
    for k in range(n_assets):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.04, n_bars)))
        volume = rng.uniform(1e3, 1e6, n_bars)
        if listing_gaps and k % 2 and n_bars >= 12:  # staggered listing and a delisting
            close[: rng.integers(1, n_bars // 3)] = np.nan
            close[n_bars - rng.integers(1, n_bars // 4) :] = np.nan
        frame = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": volume},
            index=idx,
        )
        frames[f"S{k}"] = frame
        funding[f"S{k}"] = pd.Series(rng.normal(0, 3e-4, n_bars), index=idx)
    return MarketData.from_frames(frames, "1d", funding=funding)


def _engines(cost: CostModel, controller_kind: str, basis: str, delay: int, limits: tuple):
    hard, soft, cooldown = limits

    def make(kernel: str) -> BacktestEngine:
        controller = None
        if controller_kind == "both":
            controller = DrawdownController(hard, soft, cooldown)
        elif controller_kind == "hard":
            controller = DrawdownController(hard, None, cooldown)
        elif controller_kind == "soft":
            controller = DrawdownController(None, soft, cooldown)
        return BacktestEngine(
            cost_model=cost,
            execution_delay=delay,
            drawdown_controller=controller,
            drawdown_basis=basis,
            kernel=kernel,
        )

    return make("python"), make("numba")


def _assert_same(a, b) -> None:  # type: ignore[no-untyped-def]
    for name in SERIES:
        np.testing.assert_allclose(
            getattr(a, name).to_numpy(), getattr(b, name).to_numpy(), rtol=1e-10, atol=1e-12,
            err_msg=name,
        )  # fmt: skip
    np.testing.assert_allclose(
        a.positions.to_numpy(), b.positions.to_numpy(), rtol=1e-10, atol=1e-12
    )
    if a.shadow_equity is None:
        assert b.shadow_equity is None
    else:
        np.testing.assert_allclose(
            a.shadow_equity.to_numpy(), b.shadow_equity.to_numpy(), rtol=1e-10
        )


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    n_bars=st.integers(5, 160),
    n_assets=st.integers(1, 7),
    listing_gaps=st.booleans(),
    delay=st.integers(0, 2),
    sqrt_impact=st.booleans(),
    fee=st.floats(0.0, 20.0),
    impact=st.floats(0.0, 1.0),
    controller_kind=st.sampled_from(["none", "both", "hard", "soft"]),
    basis=st.sampled_from(["shadow", "realized"]),
    leverage=st.sampled_from([0.5, 3.0]),
    start_share=st.floats(0.0, 0.5),
)
def test_kernel_matches_python_reference(
    seed: int,
    n_bars: int,
    n_assets: int,
    listing_gaps: bool,
    delay: int,
    sqrt_impact: bool,
    fee: float,
    impact: float,
    controller_kind: str,
    basis: str,
    leverage: float,
    start_share: float,
) -> None:
    data = _market(n_bars, n_assets, seed, listing_gaps)
    rng = np.random.default_rng(seed + 1)
    weights = pd.DataFrame(
        rng.normal(0, leverage / n_assets, (n_bars, n_assets)),
        index=data.index,
        columns=data.symbols,
    )
    cost = SquareRootImpactModel(fee, 1.5, impact) if sqrt_impact else LinearCostModel(fee, 1.5)
    python, numba = _engines(cost, controller_kind, basis, delay, (0.08, 0.03, 3))
    start = data.index[int(start_share * (n_bars - 1))]
    reference = python.run(weights, data, start=start)
    compiled = numba.run(weights, data, start=start)
    _assert_same(reference, compiled)
    if python.drawdown_controller is not None:  # same final state, hwm to float precision
        ref, got = python.drawdown_controller.state_dict(), numba.drawdown_controller.state_dict()
        assert (ref["cooldown"], ref["breaches"]) == (got["cooldown"], got["breaches"])
        assert (ref["hwm"] is None) == (got["hwm"] is None)
        if ref["hwm"] is not None:
            assert got["hwm"] == pytest.approx(ref["hwm"], rel=1e-10)


def test_kernel_handles_a_wiped_out_book(caplog: pytest.LogCaptureFixture) -> None:
    data = _market(40, 2, seed=3, listing_gaps=False)
    close = data.close.copy()
    close.iloc[20:] = close.iloc[19] * 0.01  # a 99% crash at bar 20
    crashed = MarketData(close, close, close, close, data.volume, "1d", funding=data.funding)
    weights = pd.DataFrame(5.0, index=data.index, columns=data.symbols)  # 10x gross long
    python, numba = _engines(LinearCostModel(4, 1.5), "both", "shadow", 0, (0.3, 0.15, 5))
    reference = python.run(weights, crashed)
    with caplog.at_level("ERROR"):
        compiled = numba.run(weights, crashed)
    assert (reference.equity == 0).any(), "scenario must wipe the book out"
    _assert_same(reference, compiled)
    assert "wiped out" in caplog.text


def test_reference_configs_give_identical_portfolios(market: MarketData) -> None:
    from alphaforge.portfolio import PortfolioConstructor
    from alphaforge.strategies import build_strategy

    signals = {
        n: build_strategy(n).generate_signals(market) for n in ("ts_momentum", "xs_momentum")
    }
    weights = PortfolioConstructor().build(signals, market).weights
    for model in (LinearCostModel(), SquareRootImpactModel()):
        python, numba = _engines(model, "both", "shadow", 1, (0.30, 0.15, 20))
        _assert_same(python.run(weights, market), numba.run(weights, market))


class _CustomCost(CostModel):
    def compute(self, trades, nav, volatility, adv):  # type: ignore[no-untyped-def]
        return np.abs(trades) * 1e-3


def test_dispatch_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    assert BacktestEngine(kernel="auto")._use_kernel()
    assert not BacktestEngine(kernel="python")._use_kernel()
    assert not BacktestEngine(cost_model=_CustomCost())._use_kernel(), "custom model -> Python"
    with pytest.raises(ValueError, match="built-in"):
        BacktestEngine(cost_model=_CustomCost(), kernel="numba")._use_kernel()
    with pytest.raises(ValueError, match="kernel must be"):
        BacktestEngine(kernel="rust")
    monkeypatch.setattr(_kernel, "AVAILABLE", False)
    assert not BacktestEngine(kernel="auto")._use_kernel(), "no numba -> Python fallback"
    with pytest.raises(ImportError, match="alphaforge\\[fast\\]"):
        BacktestEngine(kernel="numba")._use_kernel()
