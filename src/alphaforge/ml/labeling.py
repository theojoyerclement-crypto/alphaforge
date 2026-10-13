"""Supervised labels for financial ML (Lopez de Prado, *Advances in Financial ML*, ch. 3).

Every label carries its end time ``t1``: the bar at which its outcome became known. Purged
cross-validation uses ``t1`` to drop training samples whose outcome overlaps the test fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_horizon_labels(
    close: pd.DataFrame, horizon: int, threshold: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sign of the ``horizon``-bar forward return (0 inside ``+/-threshold``).

    Returns:
        ``(labels, forward_returns)``; both ``NaN`` where the outcome is not yet known.
    """
    fwd = close.shift(-horizon) / close - 1.0
    labels = np.sign(fwd).where(fwd.abs() > threshold, 0.0).where(fwd.notna())
    return labels, fwd


def ewma_volatility(
    close: pd.Series | pd.DataFrame, halflife: int = 20
) -> pd.Series | pd.DataFrame:
    """Per-bar EWMA volatility of simple returns (known at the bar's close)."""
    returns = close / close.shift(1) - 1.0
    return returns.ewm(halflife=halflife, min_periods=halflife).std()


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DatetimeIndex,
    volatility: pd.Series,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 10,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Triple-barrier labelling.

    For each event at ``t0`` a position is opened at close ``t0``. It exits at the first of:
    profit-taking at ``+pt_mult * vol``, stop-loss at ``-sl_mult * vol``, or the vertical barrier
    after ``max_holding`` bars. Events whose path is cut short by the end of the data and touch
    no barrier are dropped (their outcome is unknown; keeping them would leak).

    Args:
        close: Price series of one asset.
        events: Event start times (subset of ``close.index``).
        volatility: Per-bar return volatility used to size the barriers.
        pt_mult, sl_mult: Barrier widths in volatility units (0 disables a barrier).
        max_holding: Vertical barrier in bars.
        side: Optional position side per event (+1/-1). When given, labels are *meta-labels*:
            1 if the side's trade was profitable, else 0.

    Returns:
        Frame indexed by event time with ``t1``, ``ret`` (side-adjusted), ``label``, ``barrier``.
    """
    prices = close.to_numpy(dtype=float)
    index = close.index
    n = len(prices)
    positions = index.get_indexer(events)
    vol = volatility.reindex(events).to_numpy(dtype=float)
    sides = np.ones(len(events)) if side is None else side.reindex(events).to_numpy(dtype=float)
    valid = (positions >= 0) & np.isfinite(vol) & (vol > 0) & np.isfinite(sides) & (sides != 0)
    valid[valid] &= np.isfinite(prices[positions[valid]])
    pos, vol, sides = positions[valid], vol[valid], sides[valid]
    if max_holding < 1:
        pos = pos[:0]
        max_holding = 1

    # paths[k, j]: side-adjusted return from close pos[k] to close pos[k] + 1 + j. Bars past the
    # end of the data are NaN, which never touch a barrier (vectorised over all events at once).
    padded = np.concatenate([prices[1:], np.full(max_holding, np.nan)])
    window = np.lib.stride_tricks.sliding_window_view(padded, max_holding)[pos]
    with np.errstate(invalid="ignore"):
        paths = (window / prices[pos, None] - 1.0) * sides[:, None]
        upper = pt_mult * vol if pt_mult > 0 else np.full(len(pos), np.inf)
        lower = -sl_mult * vol if sl_mult > 0 else np.full(len(pos), -np.inf)
        hit_up = paths >= upper[:, None]
        hit_dn = paths <= lower[:, None]
    never = max_holding  # sentinel: barrier not touched within the window
    first_up = np.where(hit_up.any(axis=1), hit_up.argmax(axis=1), never)
    first_dn = np.where(hit_dn.any(axis=1), hit_dn.argmax(axis=1), never)

    touched = (first_up < never) | (first_dn < never)
    complete = pos + max_holding <= n - 1  # vertical barrier inside the data
    keep = touched | complete  # untouched and cut short by the data end: outcome unknown
    j = np.where(touched, np.minimum(first_up, first_dn), max_holding - 1)
    barrier = np.where(~touched, "time", np.where(first_up <= first_dn, "profit", "stop"))
    ret = paths[np.arange(len(pos)), j]
    keep &= np.isfinite(ret)

    pos, j, ret, barrier = pos[keep], j[keep], ret[keep], barrier[keep]
    label = (ret > 0).astype(float) if side is not None else np.sign(ret)
    out = pd.DataFrame(
        {"t0": index[pos], "t1": index[pos + 1 + j], "ret": ret, "label": label, "barrier": barrier}
    )
    return out.set_index("t0")
