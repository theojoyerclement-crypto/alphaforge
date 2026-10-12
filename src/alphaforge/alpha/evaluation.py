"""Alpha evaluation: information coefficients, decay, quantile spreads, turnover.

Forward returns are the *only* place in the framework where future data is used on purpose,
and only to score signals, never as inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from alphaforge.data.market_data import MarketData
from alphaforge.features.transforms import cs_demean, cs_rank, normalize_gross


def forward_returns(close: pd.DataFrame, horizon: int, delay: int = 0) -> pd.DataFrame:
    """Return from close ``t+delay`` to close ``t+delay+horizon`` (aligned on ``t``)."""
    return close.shift(-(horizon + delay)) / close.shift(-delay) - 1.0


def cross_sectional_ic(
    signal: pd.DataFrame,
    target: pd.DataFrame,
    method: str = "spearman",
    min_assets: int = 3,
) -> pd.Series:
    """Per-bar cross-sectional correlation between ``signal`` and ``target``.

    Args:
        method: ``"spearman"`` (rank IC, robust default) or ``"pearson"``.
        min_assets: Bars with fewer jointly valid assets yield ``NaN``.
    """
    s, y = signal.align(target, join="inner")
    mask = s.notna() & y.notna()
    s, y = s.where(mask), y.where(mask)
    if method == "spearman":
        s, y = s.rank(axis=1), y.rank(axis=1)
    elif method != "pearson":
        raise ValueError("method must be 'spearman' or 'pearson'")
    sd = s.sub(s.mean(axis=1), axis=0)
    yd = y.sub(y.mean(axis=1), axis=0)
    denom = np.sqrt((sd**2).sum(axis=1) * (yd**2).sum(axis=1)).replace(0.0, np.nan)
    ic = (sd * yd).sum(axis=1) / denom
    ic[mask.sum(axis=1) < min_assets] = np.nan
    return ic.rename("ic")


def newey_west_tstat(series: pd.Series, lags: int) -> float:
    """t-statistic of the mean with a Newey-West HAC standard error.

    Overlapping forward returns make consecutive ICs autocorrelated; a plain t-stat would
    overstate significance by roughly ``sqrt(horizon)``.
    """
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        s += 2.0 * weight * (e[lag:] @ e[:-lag]) / n
    if s <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(s / n))


def rank_autocorrelation(signal: pd.DataFrame, lag: int = 1) -> float:
    """Mean cross-sectional rank correlation of the signal with itself ``lag`` bars ago.

    Close to 1 means a slow signal (cheap to trade); close to 0 means high turnover.
    """
    return float(cross_sectional_ic(signal, signal.shift(lag)).mean())


def quantile_returns(signal: pd.DataFrame, fwd: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """Mean forward return of each cross-sectional signal quantile, per bar."""
    pct = signal.rank(axis=1, pct=True)
    bucket = np.ceil(pct * n_quantiles).clip(1, n_quantiles)
    return pd.DataFrame({q: fwd.where(bucket == q).mean(axis=1) for q in range(1, n_quantiles + 1)})


def long_short_returns(signal: pd.DataFrame, returns: pd.DataFrame, lag: int = 1) -> pd.Series:
    """Gross, cost-free returns of a rank-weighted dollar-neutral portfolio on ``signal``.

    Weights formed at close ``t`` earn returns from bar ``t + lag`` onwards.
    """
    weights = normalize_gross(cs_demean(cs_rank(signal)).fillna(0.0))
    return (weights.shift(lag) * returns).sum(axis=1, min_count=1).rename("long_short")


@dataclass(frozen=True)
class AlphaEvaluation:
    """Summary statistics of one alpha."""

    name: str
    ic_mean: float
    ic_std: float
    icir: float
    icir_annualized: float
    ic_tstat: float
    ic_pvalue: float
    ic_hit_rate: float
    rank_autocorr: float
    long_short_sharpe: float
    long_short_return_ann: float
    quantile_spread: float
    n_obs: int
    ic_decay: dict[int, float] = field(default_factory=dict)
    quantile_means: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_alpha(
    name: str,
    values: pd.DataFrame,
    data: MarketData,
    horizons: Sequence[int] = (1, 5, 10, 20),
    n_quantiles: int = 3,
) -> tuple[AlphaEvaluation, pd.Series, pd.Series]:
    """Evaluate an alpha's predictive power.

    Returns:
        ``(evaluation, ic_series_at_first_horizon, long_short_returns)``.
    """
    ppy = data.periods_per_year
    horizons = sorted(horizons)
    ic_by_h = {h: cross_sectional_ic(values, forward_returns(data.close, h)) for h in horizons}
    h0 = horizons[0]
    ic = ic_by_h[h0].dropna()
    ic_mean = float(ic.mean()) if len(ic) else float("nan")
    ic_std = float(ic.std()) if len(ic) > 1 else float("nan")
    icir = ic_mean / ic_std if ic_std and np.isfinite(ic_std) and ic_std > 0 else float("nan")
    tstat = newey_west_tstat(ic, lags=max(h0, 5))
    pvalue = float(2 * (1 - stats.norm.cdf(abs(tstat)))) if np.isfinite(tstat) else float("nan")

    ls = long_short_returns(values, data.returns()).dropna()
    ls_std = ls.std()
    ls_sharpe = float(ls.mean() / ls_std * np.sqrt(ppy)) if ls_std > 0 else float("nan")

    q = quantile_returns(values, forward_returns(data.close, h0), n_quantiles).mean()
    evaluation = AlphaEvaluation(
        name=name,
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        icir_annualized=icir * np.sqrt(ppy / h0) if np.isfinite(icir) else float("nan"),
        ic_tstat=tstat,
        ic_pvalue=pvalue,
        ic_hit_rate=float((ic > 0).mean()) if len(ic) else float("nan"),
        rank_autocorr=rank_autocorrelation(values),
        long_short_sharpe=ls_sharpe,
        long_short_return_ann=float(ls.mean() * ppy),
        quantile_spread=float(q.iloc[-1] - q.iloc[0]),
        n_obs=len(ic),
        ic_decay={h: float(s.mean()) for h, s in ic_by_h.items()},
        quantile_means={int(k): float(v) for k, v in q.items()},
    )
    return evaluation, ic_by_h[h0], ls
