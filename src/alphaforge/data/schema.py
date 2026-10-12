"""Canonical data schema and validation."""

from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


class DataValidationError(ValueError):
    """Raised when a frame violates the OHLCV schema."""


def validate_ohlcv(frame: pd.DataFrame, symbol: str = "") -> None:
    """Assert that ``frame`` is a clean single-symbol OHLCV table.

    Checks: required columns, tz-aware UTC ``DatetimeIndex``, strictly increasing timestamps,
    positive prices, non-negative volume, and ``low <= open, close <= high``.

    Raises:
        DataValidationError: on the first violated rule.
    """
    tag = f"[{symbol}] " if symbol else ""
    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise DataValidationError(f"{tag}missing columns {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataValidationError(f"{tag}index must be a DatetimeIndex")
    if frame.index.tz is None or str(frame.index.tz) != "UTC":
        raise DataValidationError(f"{tag}index must be tz-aware UTC")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise DataValidationError(f"{tag}index must be strictly increasing")

    valid = frame.dropna(subset=list(PRICE_COLUMNS))
    if (valid[list(PRICE_COLUMNS)] <= 0).any().any():
        raise DataValidationError(f"{tag}non-positive prices")
    if (frame["volume"].dropna() < 0).any():
        raise DataValidationError(f"{tag}negative volume")
    eps = 1e-12
    bad = (valid["high"] + eps < valid[["open", "close", "low"]].max(axis=1)) | (
        valid["low"] - eps > valid[["open", "close", "high"]].min(axis=1)
    )
    if bad.any():
        raise DataValidationError(f"{tag}{int(bad.sum())} bars violate low <= open,close <= high")
