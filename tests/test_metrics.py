from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.analytics import (
    cagr,
    compute_metrics,
    drawdown_table,
    max_drawdown,
    max_drawdown_duration,
    monthly_returns,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)


def _series(values: list[float]) -> pd.Series:
    return pd.Series(
        values, index=pd.date_range("2023-01-01", periods=len(values), freq="1D", tz="UTC")
    )


def test_sharpe_and_sortino_hand_computed() -> None:
    r = _series([0.01, -0.01, 0.02, 0.0])
    expected = np.mean(r) / np.std(r, ddof=1) * np.sqrt(365)
    assert sharpe_ratio(r, 365) == pytest.approx(expected)
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert sortino_ratio(r, 365) == pytest.approx(np.mean(r) / downside * np.sqrt(365))


def test_drawdown_metrics() -> None:
    r = _series([0.10, -0.50, 0.20, 1.0, -0.1])
    # equity 1.1, 0.55, 0.66, 1.32, 1.188 -> worst dd = 0.55/1.1 - 1
    assert max_drawdown(r) == pytest.approx(-0.5)
    assert max_drawdown_duration(r) == 2
    table = drawdown_table(r)
    assert table.iloc[0]["depth"] == pytest.approx(-0.5)
    assert table.iloc[0]["recovery"] == r.index[3]


def test_profit_factor_and_cagr() -> None:
    r = _series([0.02, -0.01, 0.03, -0.02])
    assert profit_factor(r) == pytest.approx(0.05 / 0.03)
    flat_year = pd.Series([0.0] * 365)
    assert cagr(flat_year, 365) == pytest.approx(0.0)
    doubling = pd.Series([2 ** (1 / 365) - 1] * 365)
    assert cagr(doubling, 365) == pytest.approx(1.0)


def test_monthly_returns_compound() -> None:
    r = pd.Series(0.01, index=pd.date_range("2023-01-01", "2023-02-28", freq="1D", tz="UTC"))
    table = monthly_returns(r)
    assert table.loc[2023, 1] == pytest.approx(1.01**31 - 1)
    assert table.loc[2023, 2] == pytest.approx(1.01**28 - 1)


def test_compute_metrics_complete() -> None:
    rng = np.random.default_rng(0)
    r = _series(list(rng.normal(0.001, 0.02, 500)))
    m = compute_metrics(r, 365, turnover=r.abs(), gross_leverage=r.abs() + 1, costs=r.abs() * 0.001)
    d = m.to_dict()
    assert d["n_periods"] == 500
    assert all(v is not None for v in d.values())
    assert m.cvar_95 <= m.var_95 < 0
