"""Purged and embargoed cross-validation for overlapping financial labels.

Standard K-fold leaks in finance twice over: (1) a training label whose window overlaps the
test period encodes test-period prices; (2) serial correlation lets samples right after the
test fold "remember" it. Purging removes (1), the embargo removes (2).

Reference: Lopez de Prado (2018), *Advances in Financial Machine Learning*, ch. 7 and 12.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np
import pandas as pd


def _times(t0: pd.Index | np.ndarray, t1: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(pd.DatetimeIndex(t0).asi8 if isinstance(t0, pd.Index) else t0)
    end = np.asarray(pd.DatetimeIndex(np.asarray(t1)).asi8)
    if start.shape != end.shape:
        raise ValueError("t0 and t1 must have the same length")
    if (end < start).any():
        raise ValueError("t1 must not precede t0")
    return start, end


def _purge(
    start: np.ndarray, end: np.ndarray, test_mask: np.ndarray, embargo_ns: int
) -> np.ndarray:
    """Training mask: not in test, no label overlap with any test block, outside embargo."""
    train = ~test_mask
    test_idx = np.flatnonzero(test_mask)
    # contiguous test blocks (in sample order)
    breaks = np.flatnonzero(np.diff(test_idx) > 1)
    for block in np.split(test_idx, breaks + 1):
        if block.size == 0:
            continue
        block_start = start[block].min()
        block_end = end[block].max()
        overlap = (start <= block_end + embargo_ns) & (end >= block_start)
        train &= ~overlap
    return train


class PurgedKFold:
    """K-fold over time with purging and embargo.

    Samples must be sorted by start time. With pooled panels (several assets per timestamp),
    folds are formed on *unique timestamps* so one bar never straddles train and test.

    Args:
        n_splits: Number of folds.
        embargo_pct: Embargo length as a fraction of the total time span.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self, t0: pd.Index | np.ndarray, t1: pd.Series | np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_positions, test_positions)``."""
        start, end = _times(t0, t1)
        if (np.diff(start) < 0).any():
            raise ValueError("Samples must be sorted by t0")
        unique = np.unique(start)
        embargo_ns = int((end.max() - start.min()) * self.embargo_pct)
        for chunk in np.array_split(unique, self.n_splits):
            test_mask = (start >= chunk[0]) & (start <= chunk[-1])
            train_mask = _purge(start, end, test_mask, embargo_ns)
            yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)


class CombinatorialPurgedKFold:
    """Combinatorial purged CV (CPCV): every choice of ``n_test_groups`` among ``n_groups``.

    Produces many backtest paths instead of one, giving a distribution of out-of-sample
    performance rather than a single, possibly lucky, number.
    """

    def __init__(
        self, n_groups: int = 6, n_test_groups: int = 2, embargo_pct: float = 0.01
    ) -> None:
        if not 1 <= n_test_groups < n_groups:
            raise ValueError("Need 1 <= n_test_groups < n_groups")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    @property
    def n_splits(self) -> int:
        return len(list(itertools.combinations(range(self.n_groups), self.n_test_groups)))

    @property
    def n_paths(self) -> int:
        """Number of complete backtest paths: ``C(N, k) * k / N``."""
        return self.n_splits * self.n_test_groups // self.n_groups

    def split(
        self, t0: pd.Index | np.ndarray, t1: pd.Series | np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        start, end = _times(t0, t1)
        unique = np.unique(start)
        groups = np.array_split(unique, self.n_groups)
        embargo_ns = int((end.max() - start.min()) * self.embargo_pct)
        for combo in itertools.combinations(range(self.n_groups), self.n_test_groups):
            test_mask = np.zeros(len(start), dtype=bool)
            for g in combo:
                test_mask |= (start >= groups[g][0]) & (start <= groups[g][-1])
            train_mask = _purge(start, end, test_mask, embargo_ns)
            yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)
