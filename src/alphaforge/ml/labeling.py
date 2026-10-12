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

    rows: list[tuple[pd.Timestamp, pd.Timestamp, float, float, str]] = []
    for i, p in enumerate(positions):
        if p < 0 or not np.isfinite(vol[i]) or vol[i] <= 0 or not np.isfinite(prices[p]):
            continue
        s = sides[i]
        if not np.isfinite(s) or s == 0:
            continue
        end = p + max_holding
        path = (prices[p + 1 : min(end, n - 1) + 1] / prices[p] - 1.0) * s
        if len(path) == 0:
            continue
        upper = pt_mult * vol[i] if pt_mult > 0 else np.inf
        lower = -sl_mult * vol[i] if sl_mult > 0 else -np.inf
        hits_up = np.flatnonzero(path >= upper)
        hits_dn = np.flatnonzero(path <= lower)
        first_up = hits_up[0] if hits_up.size else np.inf
        first_dn = hits_dn[0] if hits_dn.size else np.inf
        if np.isinf(first_up) and np.isinf(first_dn):
            if end > n - 1:
                continue  # incomplete horizon, unknown outcome
            j, barrier = len(path) - 1, "time"
        elif first_up <= first_dn:
            j, barrier = int(first_up), "profit"
        else:
            j, barrier = int(first_dn), "stop"
        ret = path[j]
        if not np.isfinite(ret):
            continue
        label = float(ret > 0) if side is not None else float(np.sign(ret))
        rows.append((index[p], index[p + 1 + j], float(ret), label, barrier))

    out = pd.DataFrame(rows, columns=["t0", "t1", "ret", "label", "barrier"])
    return out.set_index("t0")
