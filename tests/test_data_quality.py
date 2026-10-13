"""Data-quality audit, point-in-time universe, provenance and data-source failures."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from alphaforge.config.schema import DataConfig, QualityConfig, SyntheticConfig, UniverseConfig
from alphaforge.data import (
    DataQualityError,
    MarketData,
    assess_quality,
    load_dataset,
    universe_mask,
)
from alphaforge.data.quality import enforce_quality
from alphaforge.data.sources.base import DataSourceError
from alphaforge.portfolio import PortfolioConstructor
from alphaforge.strategies import build_strategy
from tests.conftest import perturb_future


def _with_close(data: MarketData, close: pd.DataFrame, volume: pd.DataFrame) -> MarketData:
    return MarketData(
        open=close, high=close, low=close, close=close, volume=volume, timeframe=data.timeframe
    )


def _flags(report_table: pd.DataFrame, symbol: str) -> set[str]:
    return set(filter(None, str(report_table.loc[symbol, "flags"]).split(",")))


# -- quality ---------------------------------------------------------------------------------


def test_clean_synthetic_panel_passes(market: MarketData) -> None:
    report = assess_quality(market)
    assert report.passed, report.summary()
    assert set(report.table.index) == set(market.symbols)


def test_quality_flags_each_failure_mode(market: MarketData) -> None:
    close, volume = market.close.copy(), market.volume.copy()
    close.iloc[300:310, 0] = np.nan  # 10-bar interior hole
    close.iloc[400:408, 1] = close.iloc[399, 1]  # dead feed: unchanged price ...
    volume.iloc[400:408, 1] = 0.0  # ... on zero volume
    close.iloc[-20:, 2] = np.nan  # stops trading before the panel end
    report = assess_quality(_with_close(market, close, volume), QualityConfig())

    a, b, c = market.symbols[:3]
    assert "long_gap" in _flags(report.table, a)
    assert "stale" in _flags(report.table, b)
    assert "ended_early" in _flags(report.table, c)
    assert not report.passed and set(report.errors) == {a, b}  # ended_early is info only
    with pytest.raises(DataQualityError, match="long_gap"):
        enforce_quality(report, "error")
    enforce_quality(report, "warn")  # warn never raises


def test_funding_gap_is_detected(market: MarketData) -> None:
    symbol = market.symbols[0]
    listed = market.close[symbol].dropna().index
    events = pd.Series(1e-4, index=pd.date_range(listed[0], listed[-1], freq="8h"))
    healthy = assess_quality(market, funding_events={symbol: events})
    assert healthy.table.loc[symbol, "funding_coverage"] > 0.99

    holed = events.drop(events.index[100:400])  # 100 days without a settlement
    report = assess_quality(market, funding_events={symbol: holed})
    span_days = (listed[-1] - listed[0]).days
    assert report.table.loc[symbol, "funding_coverage"] == pytest.approx(
        1 - 100 / span_days, abs=0.01
    )
    assert "funding_gap" in _flags(report.table, symbol)


def test_load_dataset_enforces_policy_and_keeps_old_api() -> None:
    from alphaforge.data import load_market_data

    config = DataConfig(
        symbols=["A", "B", "C"],
        synthetic=SyntheticConfig(n_bars=200),
        quality=QualityConfig(on_fail="error", spike_mad=1e6),
    )
    dataset = load_dataset(config)
    assert dataset.quality.passed and dataset.sources == []  # synthetic: no stored files
    data, cleaning = load_market_data(config)
    pd.testing.assert_frame_equal(data.close, dataset.data.close)
    assert len(cleaning) == 3


# -- point-in-time universe ------------------------------------------------------------------


def test_default_universe_is_every_listed_bar(market: MarketData) -> None:
    pd.testing.assert_frame_equal(universe_mask(market, UniverseConfig()), market.close.notna())


def test_universe_respects_top_n_and_listing_age(market: MarketData) -> None:
    cfg = UniverseConfig(top_n=3, rank_lookback=20, min_history=50)
    mask = universe_mask(market, cfg)
    assert (mask.sum(axis=1) <= 3).all() and (mask.sum(axis=1).iloc[-100:] == 3).all()
    age = market.close.notna().cumsum()
    assert not (mask & (age <= 50)).any().any(), "member before min_history bars of trading"
    assert not (mask & market.close.isna()).any().any()


def test_universe_holds_membership_between_reviews(market: MarketData) -> None:
    mask = universe_mask(market, UniverseConfig(top_n=4, rebalance_every=10)).astype(int)
    changes = mask.diff().abs().sum(axis=1)
    listed_changes = market.close.notna().astype(int).diff().abs().sum(axis=1)
    off_review = np.arange(len(mask)) % 10 != 0
    assert (changes[off_review] <= listed_changes[off_review]).all()


def test_universe_mask_is_causal(market: MarketData) -> None:
    cfg = UniverseConfig(top_n=4, rank_lookback=30, min_history=20, rebalance_every=5)
    cutoff = 600
    base = universe_mask(market, cfg)
    scrambled = universe_mask(perturb_future(market, cutoff), cfg)
    pd.testing.assert_frame_equal(base.iloc[:cutoff], scrambled.iloc[:cutoff])


def test_constructor_never_holds_non_members(market: MarketData) -> None:
    universe = UniverseConfig(top_n=3, min_history=30)
    constructor = PortfolioConstructor(universe=universe)
    signals = {
        n: build_strategy(n).generate_signals(market) for n in ("ts_momentum", "xs_momentum")
    }
    targets = constructor.build(signals, market)
    outside = ~universe_mask(market, universe)
    assert (targets.weights.where(outside, 0.0) == 0.0).all().all()
    for sleeve in targets.sleeve_weights.values():
        assert (sleeve.where(outside, 0.0).fillna(0.0) == 0.0).all().all()
    assert targets.weights.abs().sum(axis=1).iloc[-100:].gt(0).all()


# -- config validation -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"symbols": ["A", "A"]}, "duplicate symbols"),
        ({"start": "2022-01-01", "end": "2021-01-01"}, "must be before"),
        ({"start": "last tuesday"}, "is not a date"),
        ({"universe": {"top_n": 5}}, "exceeds"),
    ],
)
def test_data_config_rejects_inconsistent_settings(overrides: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        DataConfig.model_validate({"symbols": ["A", "B"], **overrides})


# -- data-source failures --------------------------------------------------------------------


def _fake_ccxt() -> types.SimpleNamespace:
    class BaseError(Exception):
        pass

    class NetworkError(BaseError):
        pass

    class ExchangeNotAvailable(NetworkError):  # noqa: N818 (mirrors the ccxt name)
        pass

    class ExchangeError(BaseError):
        pass

    return types.SimpleNamespace(
        NetworkError=NetworkError,
        ExchangeNotAvailable=ExchangeNotAvailable,
        ExchangeError=ExchangeError,
    )


def test_ccxt_failures_become_actionable_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from alphaforge.data.sources import ccxt_source
    from alphaforge.data.sources.ccxt_source import CCXTSource

    monkeypatch.setattr(ccxt_source.time, "sleep", lambda _: None)
    fake = _fake_ccxt()
    source = object.__new__(CCXTSource)  # no ccxt install or network needed
    source._ccxt, source.name, source.max_retries = fake, "ccxt_test_swap", 3

    calls = []

    def flaky() -> list[int]:
        calls.append(1)
        raise fake.NetworkError("timed out")

    with pytest.raises(DataSourceError, match=r"failed after 3 attempts.*data.source: parquet"):
        source._call(flaky)
    assert len(calls) == 3

    def rejected() -> None:
        raise fake.ExchangeError("binance does not have market symbol FOO/USDT")

    with pytest.raises(DataSourceError, match=r"rejected.*FOO/USDT"):
        source._call(rejected)
