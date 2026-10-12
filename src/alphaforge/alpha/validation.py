"""Statistical validation under multiple testing.

Testing many alphas guarantees that some look good by chance. These tools answer
"is this Sharpe/IC distinguishable from luck, *given how many things I tried*?"

References:
    Bailey & Lopez de Prado (2012), The Sharpe Ratio Efficient Frontier.
    Bailey & Lopez de Prado (2014), The Deflated Sharpe Ratio.
    Benjamini & Hochberg (1995), Controlling the False Discovery Rate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

_EULER_GAMMA = 0.5772156649015329


def _moments(returns: pd.Series) -> tuple[float, int, float, float]:
    r = returns.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return float("nan"), n, float("nan"), float("nan")
    sr = r.mean() / r.std(ddof=1)
    return float(sr), n, float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > ``sr_benchmark``), correcting for skew and fat tails.

    Args:
        returns: Per-period strategy returns.
        sr_benchmark: Per-period (not annualised) Sharpe threshold.
    """
    sr, n, skew, kurt = _moments(returns)
    if not np.isfinite(sr):
        return float("nan")
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` skill-less strategies."""
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - _EULER_GAMMA) * a + _EULER_GAMMA * b))


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int, sr_variance: float) -> float:
    """PSR against the Sharpe a lucky best-of-``n_trials`` would achieve by chance.

    Args:
        returns: Per-period returns of the selected strategy.
        n_trials: Number of (effectively independent) configurations tried.
        sr_variance: Variance of the per-period Sharpe ratios across those trials.
    """
    return probabilistic_sharpe_ratio(returns, expected_max_sharpe(n_trials, sr_variance))


def minimum_track_record_length(
    returns: pd.Series, sr_benchmark: float = 0.0, confidence: float = 0.95
) -> float:
    """Bars needed for the observed Sharpe to beat ``sr_benchmark`` at ``confidence``."""
    sr, _, skew, kurt = _moments(returns)
    if not np.isfinite(sr) or sr <= sr_benchmark:
        return float("inf")
    z = stats.norm.ppf(confidence)
    return float(
        1.0 + (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) * (z / (sr - sr_benchmark)) ** 2
    )


def benjamini_hochberg(pvalues: Sequence[float], fdr: float = 0.10) -> np.ndarray:
    """Boolean mask of discoveries controlling the false discovery rate at ``fdr``.

    ``NaN`` p-values are never discoveries.
    """
    p = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(p)
    m = int(valid.sum())
    result = np.zeros(len(p), dtype=bool)
    if m == 0:
        return result
    idx = np.flatnonzero(valid)
    order = idx[np.argsort(p[idx])]
    thresholds = fdr * np.arange(1, m + 1) / m
    passed = p[order] <= thresholds
    if passed.any():
        cutoff = np.max(np.flatnonzero(passed))
        result[order[: cutoff + 1]] = True
    return result
