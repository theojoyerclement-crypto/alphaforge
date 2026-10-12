"""Panel transforms shared by features, alphas, strategies and portfolio construction.

All transforms are row-wise (cross-sectional) or trailing (time-series): none look ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cs_rank(panel: pd.DataFrame, centered: bool = True) -> pd.DataFrame:
    """Cross-sectional percentile rank per row, in ``(0, 1]`` or centred to ``(-0.5, 0.5]``."""
    ranks = panel.rank(axis=1, pct=True)
    return ranks - 0.5 if centered else ranks


def cs_zscore(panel: pd.DataFrame, clip: float | None = 3.0) -> pd.DataFrame:
    """Cross-sectional z-score per row (optionally clipped)."""
    std = panel.std(axis=1).replace(0.0, np.nan)
    z = panel.sub(panel.mean(axis=1), axis=0).div(std, axis=0)
    return z.clip(-clip, clip) if clip is not None else z


def cs_demean(panel: pd.DataFrame) -> pd.DataFrame:
    """Subtract the cross-sectional mean of each row."""
    return panel.sub(panel.mean(axis=1), axis=0)


def ts_zscore(panel: pd.DataFrame, lookback: int, min_periods: int | None = None) -> pd.DataFrame:
    """Trailing time-series z-score over ``lookback`` bars."""
    mp = min_periods or lookback
    roll = panel.rolling(lookback, min_periods=mp)
    return (panel - roll.mean()) / roll.std().replace(0.0, np.nan)


def normalize_gross(weights: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """Scale each row so that ``sum(|w|) == gross`` (all-zero rows stay zero)."""
    total = weights.abs().sum(axis=1).replace(0.0, np.nan)
    return weights.div(total, axis=0).mul(gross).fillna(0.0)


def long_short_quantiles(
    score: pd.DataFrame, quantile: float = 0.3, min_assets: int = 4
) -> pd.DataFrame:
    """Dollar-neutral quantile portfolio: long the top ``quantile``, short the bottom.

    Each leg is equally weighted and sums to 0.5 in absolute value (gross = 1).
    Rows with fewer than ``min_assets`` valid scores are flat.
    """
    if not 0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    valid = score.notna()
    pct = score.rank(axis=1, pct=True)
    longs = ((pct > 1 - quantile) & valid).astype(float)
    shorts = ((pct <= quantile) & valid).astype(float)
    w = 0.5 * longs.div(longs.sum(axis=1).replace(0, np.nan), axis=0) - 0.5 * shorts.div(
        shorts.sum(axis=1).replace(0, np.nan), axis=0
    )
    w = w.fillna(0.0)
    w[valid.sum(axis=1) < min_assets] = 0.0
    return w


def scale_to_unit_max(weights: pd.DataFrame) -> pd.DataFrame:
    """Scale each row so its largest absolute entry is 1 (all-zero rows stay zero)."""
    peak = weights.abs().max(axis=1).replace(0.0, np.nan)
    return weights.div(peak, axis=0).fillna(0.0)


def panel_to_long(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack aligned wide panels into a long frame indexed by ``(timestamp, symbol)``.

    Version-independent alternative to ``DataFrame.stack`` (whose NaN semantics changed in
    pandas 3). Rows are *not* dropped; callers decide how to treat missing values.
    """
    if not panels:
        raise ValueError("No panels to stack")
    first = next(iter(panels.values()))
    index = pd.MultiIndex.from_product([first.index, first.columns], names=["timestamp", "symbol"])
    data = {name: panel.reindex_like(first).to_numpy().ravel() for name, panel in panels.items()}
    return pd.DataFrame(data, index=index)


def long_to_panel(series: pd.Series, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    """Inverse of :func:`panel_to_long` for one column."""
    wide = series.unstack("symbol")
    return wide.reindex(index=index, columns=columns)
