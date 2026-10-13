"""Bootstrap Sharpe intervals: calibration, pairing and determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.analytics import sharpe_difference, sharpe_interval, sharpe_ratio
from alphaforge.analytics.inference import default_block_length

PPY = 365.0


def _series(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), tz="UTC"))


def test_point_estimate_matches_published_sharpe() -> None:
    r = _series(np.random.default_rng(0).normal(0.001, 0.02, 1500))
    assert sharpe_interval(r, PPY).estimate == pytest.approx(sharpe_ratio(r, PPY))


def test_interval_covers_the_true_sharpe_about_95_percent_of_the_time() -> None:
    """Calibration on AR(1) returns with a known Sharpe (true value: 0.75 annualised)."""
    true_sharpe, n, hits, trials = 0.75, 1000, 0, 120
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        eps = rng.normal(size=n)
        x = np.empty(n)
        x[0] = eps[0]
        for t in range(1, n):  # mild autocorrelation, which the blocks have to absorb
            x[t] = 0.1 * x[t - 1] + np.sqrt(1 - 0.01) * eps[t]
        r = _series(true_sharpe / np.sqrt(PPY) * 0.02 + 0.02 * x)
        ci = sharpe_interval(r, PPY, n_boot=800, seed=seed)
        hits += ci.low <= true_sharpe <= ci.high
    assert 0.88 <= hits / trials <= 0.99


def test_paired_difference_cancels_shared_moves() -> None:
    rng = np.random.default_rng(1)
    market = rng.normal(0, 0.03, 1500)
    a = _series(market + rng.normal(0.0004, 0.005, 1500))
    b = _series(market + rng.normal(0.0, 0.005, 1500))
    paired = sharpe_difference(a, b, PPY)
    unpaired_width = 2 * 1.96 * np.sqrt(2 / (1500 / PPY))  # iid, independent-series width
    assert paired.high - paired.low < 0.5 * unpaired_width
    assert sharpe_difference(a, a, PPY).high == pytest.approx(0.0, abs=1e-12)


def test_difference_uses_common_dates_only() -> None:
    r = _series(np.random.default_rng(2).normal(0.001, 0.02, 400))
    shorter = r.iloc[100:]
    diff = sharpe_difference(r, shorter, PPY)
    assert diff.n_obs == 300 and diff.estimate == pytest.approx(0.0)


def test_deterministic_and_block_rule() -> None:
    r = _series(np.random.default_rng(3).normal(0.001, 0.02, 1931))
    assert str(sharpe_interval(r, PPY, seed=7)) == str(sharpe_interval(r, PPY, seed=7))
    assert default_block_length(1931) == 13 and sharpe_interval(r, PPY).block == 13
    with pytest.raises(ValueError):
        sharpe_interval(r.iloc[:5], PPY)
