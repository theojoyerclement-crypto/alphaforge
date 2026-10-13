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


class _FakeExchange:
    """Minimal CCXT stand-in: hourly bars up to and including the still-forming one."""

    def __init__(self, n_bars: int) -> None:
        self.has = {"fetchFundingRateHistory": False}
        now = pd.Timestamp.now(tz="UTC").floor("1h")
        opens = pd.date_range(end=now, periods=n_bars, freq="1h")
        self.rows = [[int(t.timestamp() * 1000), 1.0, 1.1, 0.9, 1.0, 5.0] for t in opens]

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> list[list[float]]:
        return [r for r in self.rows if r[0] >= since][:limit]


def test_ccxt_source_paginates_and_drops_the_forming_bar() -> None:
    from alphaforge.data.sources.ccxt_source import CCXTSource

    source = object.__new__(CCXTSource)  # skip __init__: no ccxt dependency or network
    source.exchange = _FakeExchange(n_bars=25)
    source.name, source.max_retries, source.page_limit = "fake", 1, 7

    frame = source.fetch_ohlcv(
        "BTC/USDT:USDT", "1h", start=pd.Timestamp.now(tz="UTC") - pd.Timedelta("2D")
    )
    assert len(frame) == 24, "the last, still-open bar must not be stored as if it were final"
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert frame.index[-1] + pd.Timedelta("1h") <= pd.Timestamp.now(tz="UTC")
    assert source.fetch_funding("BTC/USDT:USDT") is None


def test_dead_feed_after_delisting_becomes_untradable() -> None:
    """Binance kept serving flat, zero-volume FTM candles for 21 months after the delisting."""
    frame = _frame(60)
    frame.index = pd.date_range("2023-01-01", periods=60, freq="1D", tz="UTC")
    last = frame.iloc[39]
    frame.iloc[40:, :4] = last["close"]  # placeholder candles: O=H=L=C=last close ...
    frame.iloc[40:, 4] = 0.0  # ... on zero volume
    frame.iloc[10:12, :4] = frame.iloc[9]["close"]  # 2 quiet flat bars: a market, not a feed
    frame.iloc[10:12, 4] = 0.0
    clean, report = clean_ohlcv(frame, "1d", CleaningConfig())
    assert report.stale_removed == 20
    assert clean["close"].dropna().index[-1] == frame.index[39], "the last real bar is kept"
    assert clean.loc[frame.index[10:12], "close"].notna().all()

    kept, report = clean_ohlcv(frame, "1d", CleaningConfig(stale_bars=None))
    assert report.stale_removed == 0 and kept["close"].notna().sum() == 60
