"""Drawdown episode analysis."""

from __future__ import annotations

import pandas as pd

from alphaforge.analytics.metrics import drawdown_series


def drawdown_table(returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """The ``top_n`` deepest drawdown episodes.

    Columns: ``peak``, ``trough``, ``recovery`` (``NaT`` if not recovered), ``depth``,
    ``bars_to_trough``, ``bars_to_recovery``, ``total_bars``.
    """
    dd = drawdown_series(returns)
    underwater = dd < 0
    episodes: list[tuple[int, int | None]] = []
    in_episode = False
    start = 0
    values = dd.to_numpy()
    for i, flag in enumerate(underwater.to_numpy()):
        if flag and not in_episode:
            in_episode, start = True, i
        elif not flag and in_episode:
            episodes.append((start, i))
            in_episode = False
    if in_episode:
        episodes.append((start, None))

    rows = []
    idx = dd.index
    for s, e in episodes:
        segment = values[s:e]
        trough = s + int(segment.argmin())
        peak = max(s - 1, 0)
        rows.append(
            {
                "peak": idx[peak],
                "trough": idx[trough],
                "recovery": idx[e] if e is not None else pd.NaT,
                "depth": float(values[trough]),
                "bars_to_trough": trough - peak,
                "bars_to_recovery": (e - trough) if e is not None else None,
                "total_bars": ((e if e is not None else len(values)) - peak),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("depth").head(top_n).reset_index(drop=True)
