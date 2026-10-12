from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.backtest.cv import CombinatorialPurgedKFold, PurgedKFold


def _events(n: int = 300, horizon: int = 5, assets: int = 1) -> tuple[pd.DatetimeIndex, pd.Series]:
    times = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    t0 = times.repeat(assets)
    t1 = pd.Series(times[np.minimum(np.arange(n) + horizon, n - 1)].repeat(assets))
    return pd.DatetimeIndex(t0), t1


@pytest.mark.parametrize("assets", [1, 3])
def test_purged_kfold_has_no_label_overlap(assets: int) -> None:
    t0, t1 = _events(assets=assets)
    start, end = t0.asi8, pd.DatetimeIndex(t1).asi8
    seen_test = []
    for train, test in PurgedKFold(5, embargo_pct=0.02).split(t0, t1):
        assert len(np.intersect1d(train, test)) == 0
        test_lo, test_hi = start[test].min(), end[test].max()
        overlaps = (start[train] <= test_hi) & (end[train] >= test_lo)
        assert not overlaps.any()
        # no timestamp split between train and test (pooled panels)
        assert not np.intersect1d(start[train], start[test]).size
        seen_test.append(test)
    assert np.array_equal(np.sort(np.concatenate(seen_test)), np.arange(len(t0)))


def test_embargo_removes_samples_after_test() -> None:
    t0, t1 = _events()
    folds = list(PurgedKFold(5, embargo_pct=0.0).split(t0, t1))
    folds_emb = list(PurgedKFold(5, embargo_pct=0.05).split(t0, t1))
    assert len(folds_emb[1][0]) < len(folds[1][0])


def test_combinatorial_cv_counts() -> None:
    t0, t1 = _events()
    cv = CombinatorialPurgedKFold(n_groups=6, n_test_groups=2)
    splits = list(cv.split(t0, t1))
    assert len(splits) == cv.n_splits == 15
    assert cv.n_paths == 5
    for train, test in splits:
        assert len(np.intersect1d(train, test)) == 0


def test_rejects_bad_input() -> None:
    t0, t1 = _events()
    with pytest.raises(ValueError):
        list(PurgedKFold(3).split(t0[::-1], t1[::-1]))
    with pytest.raises(ValueError):
        PurgedKFold(1)
