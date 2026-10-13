"""Raw data providers."""

from alphaforge.data.sources.base import DataSource, DataSourceError
from alphaforge.data.sources.ccxt_source import CCXTSource
from alphaforge.data.sources.synthetic import (
    SyntheticMarket,
    SyntheticSource,
    generate_synthetic_market,
)

__all__ = [
    "CCXTSource",
    "DataSource",
    "DataSourceError",
    "SyntheticMarket",
    "SyntheticSource",
    "generate_synthetic_market",
]
