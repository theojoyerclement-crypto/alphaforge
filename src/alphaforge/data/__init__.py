"""Data layer: sources, cleaning, storage and the aligned ``MarketData`` panel."""

from alphaforge.data.cleaning import CleaningReport, aggregate_funding_to_bars, clean_ohlcv
from alphaforge.data.loader import Dataset, load_dataset, load_market_data
from alphaforge.data.market_data import MarketData
from alphaforge.data.quality import DataQualityError, QualityReport, assess_quality
from alphaforge.data.schema import OHLCV_COLUMNS, DataValidationError, validate_ohlcv
from alphaforge.data.storage import ParquetStore
from alphaforge.data.universe import universe_mask

__all__ = [
    "OHLCV_COLUMNS",
    "CleaningReport",
    "DataQualityError",
    "DataValidationError",
    "Dataset",
    "MarketData",
    "ParquetStore",
    "QualityReport",
    "aggregate_funding_to_bars",
    "assess_quality",
    "clean_ohlcv",
    "load_dataset",
    "load_market_data",
    "universe_mask",
    "validate_ohlcv",
]
