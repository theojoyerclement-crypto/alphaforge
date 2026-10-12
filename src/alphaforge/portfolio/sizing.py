"""Position sizing: volatility estimates, inverse-vol scaling, volatility targeting, Kelly."""

from __future__ import annotations

import numpy as np
import pandas as pd


def asset_volatility(
    returns: pd.DataFrame, halflife: int, periods_per_year: float, min_periods: int | None = None
) -> pd.DataFrame:
    """Annualised EWMA volatility per asset (estimate at ``t`` uses returns up to ``t``)."""
    mp = min_periods or halflife
    return returns.ewm(halflife=halflife, min_periods=mp).std() * np.sqrt(periods_per_year)


def inverse_volatility(signals: pd.DataFrame, volatility: pd.DataFrame) -> pd.DataFrame:
    """Scale signals by ``1 / vol`` so every asset contributes similar risk."""
    scaled = signals / volatility.replace(0.0, np.nan)
    return scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def ex_ante_volatility(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    halflife: int,
    periods_per_year: float,
    min_periods: int | None = None,
) -> pd.Series:
    """Annualised ex-ante portfolio volatility ``sqrt(w' Sigma w)`` with an EWMA covariance.

    The covariance at ``t`` is built from returns up to ``t`` (known at the close of ``t``) and
    is bias-corrected for the zero initialisation of the recursion. Missing returns count as
    zero (unlisted assets carry no risk; the weights are zero there anyway).
    """
    lam = 0.5 ** (1.0 / halflife)
    R = returns.reindex_like(weights).fillna(0.0).to_numpy()
    W = weights.fillna(0.0).to_numpy()
    n, k = R.shape
    cov = np.zeros((k, k))
    out = np.full(n, np.nan)
    mp = min_periods or halflife
    for t in range(n):
        r = R[t]
        cov = lam * cov + (1.0 - lam) * np.outer(r, r)
        if t + 1 >= mp:
            corrected = cov / (1.0 - lam ** (t + 1))
            w = W[t]
            out[t] = np.sqrt(max(float(w @ corrected @ w), 0.0) * periods_per_year)
    return pd.Series(out, index=weights.index, name="ex_ante_vol")


def volatility_target(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float,
    halflife: int,
    periods_per_year: float,
    max_leverage: float = 3.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Scale a book to an annualised volatility target.

    Returns:
        ``(scaled_weights, scale)``. The scale is capped so gross leverage never exceeds
        ``max_leverage``, and is zero until a risk estimate exists (no blind sizing).
    """
    vol = ex_ante_volatility(weights, returns, halflife, periods_per_year)
    scale = (target_vol / vol.replace(0.0, np.nan)).fillna(0.0)
    gross = weights.abs().sum(axis=1)
    cap = (max_leverage / gross.replace(0.0, np.nan)).fillna(0.0)
    scale = np.minimum(scale, cap).rename("vol_target_scale")
    return weights.mul(scale, axis=0), scale


def kelly_fraction(expected_return: float, variance: float, fraction: float = 0.5) -> float:
    """Fractional Kelly leverage ``fraction * mu / sigma^2``.

    Full Kelly maximises log growth but is brutally sensitive to estimation error in ``mu``;
    half Kelly keeps ~75% of the growth with half the volatility.
    """
    if variance <= 0:
        raise ValueError("variance must be positive")
    return fraction * expected_return / variance
