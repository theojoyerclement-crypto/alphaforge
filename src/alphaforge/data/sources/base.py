"""Abstract data source interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """A provider of raw per-symbol OHLCV bars and (optionally) funding events.

    Implementations return *raw* data; cleaning and alignment happen in
    :mod:`alphaforge.data.cleaning` and :mod:`alphaforge.data.loader`.
    """

    name: str = "base"

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return a frame with columns ``open, high, low, close, volume`` and a UTC index
        of bar *open* times."""

    def fetch_funding(
        self,
        symbol: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.Series | None:
        """Return funding *events* (rate per settlement, UTC-indexed), or ``None`` if the
        venue/instrument has no funding."""
        return None
