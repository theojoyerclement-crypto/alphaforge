"""Confidence intervals for Sharpe ratios and Sharpe differences (circular block bootstrap).

A point Sharpe over five years of daily crypto returns is noisy: the standard error is about
``sqrt(1 / years)`` ≈ 0.45 annualised. Comparisons such as "overlay v2 vs no overlay" or
"book vs BTC" are only meaningful if the interval of the *difference* excludes zero.

Method, fixed before looking at any result (docs/METHODOLOGY.md section 7):

* circular block bootstrap (Politis & Romano, 1992), which keeps autocorrelation and volatility
  clustering inside blocks. Block length ``ceil(n ** (1/3))``, a standard rate that is not tuned;
* 5,000 resamples, two-sided percentile intervals, fixed seed;
* differences are **paired**: both series are resampled with the same blocks on their common
  dates, so shared market moves cancel instead of inflating the interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SharpeInterval:
    """Point estimate and bootstrap interval of an annualised Sharpe (or a difference)."""

    estimate: float
    low: float
    high: float
    level: float
    n_obs: int
    block: int

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def __str__(self) -> str:
        return f"{self.estimate:+.2f} [{self.low:+.2f}, {self.high:+.2f}]"


def default_block_length(n_obs: int) -> int:
    """``ceil(n^(1/3))``: the usual rate for block bootstraps of means (Hall et al., 1995)."""
    return max(1, math.ceil(n_obs ** (1.0 / 3.0)))


def _block_indices(n: int, block: int, n_boot: int, seed: int) -> np.ndarray:
    """``(n_boot, n)`` resampling indices made of circular blocks of length ``block``."""
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, n_blocks * block)[:, :n]


def _annualised_sharpe(x: np.ndarray, periods_per_year: float) -> np.ndarray:
    """Sharpe along the last axis; NaN where the volatility is zero."""
    std = x.std(axis=-1, ddof=1)
    mean = x.mean(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(std > 0, mean / std * np.sqrt(periods_per_year), np.nan)


def _interval(point: float, draws: np.ndarray, level: float, n: int, block: int) -> SharpeInterval:
    draws = draws[np.isfinite(draws)]
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return SharpeInterval(float(point), float(low), float(high), level, n, block)


def sharpe_interval(
    returns: pd.Series,
    periods_per_year: float,
    level: float = 0.95,
    n_boot: int = 5000,
    block: int | None = None,
    seed: int = 0,
) -> SharpeInterval:
    """Bootstrap interval of the annualised Sharpe of one return series."""
    x = returns.dropna().to_numpy(dtype=float)
    if len(x) < 10:
        raise ValueError("need at least 10 observations")
    b = block or default_block_length(len(x))
    draws = _annualised_sharpe(x[_block_indices(len(x), b, n_boot, seed)], periods_per_year)
    point = float(_annualised_sharpe(x, periods_per_year))
    return _interval(point, draws, level, len(x), b)


def sharpe_difference(
    a: pd.Series,
    b: pd.Series,
    periods_per_year: float,
    level: float = 0.95,
    n_boot: int = 5000,
    block: int | None = None,
    seed: int = 0,
) -> SharpeInterval:
    """Paired bootstrap interval of ``Sharpe(a) - Sharpe(b)`` on their common dates."""
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(joined) < 10:
        raise ValueError("need at least 10 common observations")
    x = joined.to_numpy(dtype=float).T  # (2, n)
    n = x.shape[1]
    blk = block or default_block_length(n)
    idx = _block_indices(n, blk, n_boot, seed)
    draws = _annualised_sharpe(x[0][idx], periods_per_year) - _annualised_sharpe(
        x[1][idx], periods_per_year
    )
    point = float(
        _annualised_sharpe(x[0], periods_per_year) - _annualised_sharpe(x[1], periods_per_year)
    )
    return _interval(point, draws, level, n, blk)
