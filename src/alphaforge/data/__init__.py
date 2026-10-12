"""Data layer: sources, cleaning, storage and the aligned ``MarketData`` panel."""

from alphaforge.data.cleaning import CleaningReport, aggregate_funding_to_bars, clean_ohlcv
from alphaforge.data.loader import load_market_data
from alphaforge.data.market_data import MarketData
from alphaforge.data.schema import OHLCV_COLUMNS, DataValidationError, validate_ohlcv
from alphaforge.data.storage import ParquetStore

__all__ = [
    "OHLCV_COLUMNS",
    "CleaningReport",
    "DataValidationError",
    "MarketData",
    "ParquetStore",
    "aggregate_funding_to_bars",
    "clean_ohlcv",
    "load_market_data",
    "validate_ohlcv",
]
