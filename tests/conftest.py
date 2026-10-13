"""Shared fixtures: a small deterministic synthetic market."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import settings

from alphaforge.config.schema import DataConfig, SyntheticConfig
from alphaforge.data.loader import load_market_data
from alphaforge.data.market_data import MarketData

SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX"]

# Property tests explore new random cases locally, but CI must be reproducible: the same commit
# gives the same verdict. GitHub Actions sets CI=true.
settings.register_profile("ci", derandomize=True, print_blob=True)
if os.environ.get("CI"):
    settings.load_profile("ci")


@pytest.fixture(scope="session")
def market() -> MarketData:
    """8 assets x 800 daily bars with funding, basis and staggered listings."""
    config = DataConfig(
        source="synthetic",
        symbols=SYMBOLS,
        timeframe="1d",
        synthetic=SyntheticConfig(n_bars=800, start="2021-01-01"),
    )
    data, _ = load_market_data(config, seed=123)
    return data


@pytest.fixture
def constant_market() -> MarketData:
    """Two assets with hand-computable returns: A +1%/bar, B alternating +2% / -1%."""
    index = pd.date_range("2022-01-01", periods=10, freq="1D", tz="UTC", name="timestamp")
    a = 100 * 1.01 ** np.arange(10)
    b = 100 * np.cumprod([1.0] + [1.02 if i % 2 else 0.99 for i in range(1, 10)])
    frames = {
        sym: pd.DataFrame(
            {"open": p, "high": p * 1.001, "low": p * 0.999, "close": p, "volume": 1e9 / p},
            index=index,
        )
        for sym, p in {"A": a, "B": b}.items()
    }
    return MarketData.from_frames(frames, "1d")


def perturb_future(data: MarketData, cutoff: int, seed: int = 0) -> MarketData:
    """Copy of ``data`` with every panel scrambled from row ``cutoff`` onwards.

    Used by causality tests: anything computed for rows < cutoff must be unchanged.
    """
    rng = np.random.default_rng(seed)

    def scramble(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        tail = out.iloc[cutoff:].to_numpy()
        noise = np.exp(rng.normal(0.0, 0.3, size=tail.shape))
        out.iloc[cutoff:] = tail * noise
        return out

    high = scramble(data.high)
    low = scramble(data.low)
    close = scramble(data.close)
    open_ = scramble(data.open)
    # keep OHLC consistent after scrambling
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return MarketData(
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=scramble(data.volume),
        timeframe=data.timeframe,
        funding=None if data.funding is None else scramble(data.funding),
        extras={k: scramble(v) for k, v in data.extras.items()},
    )
