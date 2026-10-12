from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.config.schema import CleaningConfig, SyntheticConfig
from alphaforge.data import (
    DataValidationError,
    MarketData,
    ParquetStore,
    aggregate_funding_to_bars,
    clean_ohlcv,
    validate_ohlcv,
)
from alphaforge.data.sources.synthetic import generate_synthetic_market


def _frame(n: int = 50) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, n)))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 10.0},
        index=idx,
    )


def test_clean_removes_duplicates_and_fills_short_gaps() -> None:
    raw = _frame()
    raw = pd.concat([raw, raw.iloc[[5]]])  # duplicate
    raw = raw.drop(raw.index[[10, 11]])  # 2-bar gap
    clean, report = clean_ohlcv(raw, "1h", CleaningConfig(max_ffill_bars=3), "X")
    assert report.duplicates_removed == 1
    assert report.gaps_filled == 2
    assert len(clean) == 50
    assert clean["volume"].iloc[10] == 0.0  # synthetic flat bar carries no volume
    validate_ohlcv(clean, "X")


def test_clean_leaves_long_gaps_missing() -> None:
    raw = _frame().drop(_frame().index[10:20])
    _, report = clean_ohlcv(raw, "1h", CleaningConfig(max_ffill_bars=3))
    assert report.gaps_filled == 3
    assert report.gaps_remaining == 7


def test_clean_repairs_spike_and_fixes_ohlc() -> None:
    raw = _frame()
    raw.iloc[20, raw.columns.get_loc("close")] *= 5.0  # bad tick that reverts next bar
    raw.iloc[30, raw.columns.get_loc("high")] = raw["low"].iloc[30] * 0.5  # high < low
    clean, report = clean_ohlcv(raw, "1h")
    assert report.outliers_repaired == 1
    assert report.ohlc_fixed >= 1
    assert clean["close"].iloc[20] == pytest.approx(clean["close"].iloc[19])
    validate_ohlcv(clean)


def test_validate_rejects_naive_index() -> None:
    raw = _frame()
    raw.index = raw.index.tz_localize(None)
    with pytest.raises(DataValidationError):
        validate_ohlcv(raw)


def test_funding_aggregation_sums_events_into_bars() -> None:
    idx = pd.date_range("2023-01-01", periods=3, freq="1D", tz="UTC")
    events = pd.Series(
        [0.0001, 0.0002, 0.0003, -0.0001],
        index=pd.to_datetime(
            ["2023-01-01 00:00", "2023-01-01 08:00", "2023-01-01 16:00", "2023-01-02 08:00"],
            utc=True,
        ),
    )
    bars = aggregate_funding_to_bars(events, idx, "1d")
    assert bars.tolist() == pytest.approx([0.0006, -0.0001, 0.0])


def test_synthetic_is_deterministic_and_staggered() -> None:
    params = SyntheticConfig(n_bars=300)
    a = generate_synthetic_market(["A", "B", "C", "D"], "1d", params, seed=5)
    b = generate_synthetic_market(["A", "B", "C", "D"], "1d", params, seed=5)
    pd.testing.assert_frame_equal(a.ohlcv["A"], b.ohlcv["A"])
    assert min(len(f) for f in a.ohlcv.values()) < 300  # one late listing
    for sym, frame in a.ohlcv.items():
        validate_ohlcv(frame, sym)


def test_market_data_alignment_and_slicing(market: MarketData) -> None:
    assert market.close.shape == (800, 8)
    assert market.funding is not None and market.funding.shape == market.close.shape
    late = market.close.isna().any()
    assert late.any(), "fixture should contain a late listing"
    sub = market.iloc(100, 200)
    assert sub.n_bars == 100 and sub.index[0] == market.index[100]
    with pytest.raises(ValueError):
        MarketData(market.open, market.high, market.low, market.close.iloc[:-1], market.volume)


def test_parquet_store_roundtrip_and_merge(tmp_path) -> None:
    store = ParquetStore(tmp_path)
    frame = _frame(30)
    store.write(frame.iloc[:20], "test", "BTC/USDT:USDT", "1h")
    store.write(frame.iloc[15:], "test", "BTC/USDT:USDT", "1h")
    loaded = store.read("test", "BTC/USDT:USDT", "1h")
    pd.testing.assert_frame_equal(loaded, frame, check_freq=False)
    assert store.path("test", "BTC/USDT:USDT", "1h").with_suffix(".json").exists()
    assert store.list_symbols("test") == ["BTC-USDT_USDT"]
