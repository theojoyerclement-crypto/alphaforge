"""Timeframe conventions.

Crypto trades 24/7, so annualisation uses 365 calendar days (not 252 trading days).
Timeframes use CCXT notation (``"1h"``, ``"4h"``, ``"1d"``).
"""

from __future__ import annotations

import pandas as pd

DAYS_PER_YEAR = 365.0

_TIMEFRAME_TO_FREQ: dict[str, str] = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1D",
}


def supported_timeframes() -> list[str]:
    """Timeframes understood by the framework."""
    return list(_TIMEFRAME_TO_FREQ)


def timeframe_to_freq(timeframe: str) -> str:
    """Map a CCXT timeframe (``"4h"``) to a pandas frequency string (``"4h"``)."""
    try:
        return _TIMEFRAME_TO_FREQ[timeframe]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}; expected one of {supported_timeframes()}"
        ) from exc


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Bar duration of a timeframe."""
    return pd.Timedelta(timeframe_to_freq(timeframe))


def periods_per_year(timeframe: str) -> float:
    """Number of bars in a 365-day year for ``timeframe``."""
    return float(pd.Timedelta(days=DAYS_PER_YEAR) / timeframe_to_timedelta(timeframe))


def bars_per_day(timeframe: str) -> float:
    """Number of bars per calendar day."""
    return float(pd.Timedelta(days=1) / timeframe_to_timedelta(timeframe))


def to_utc(timestamp: str | pd.Timestamp) -> pd.Timestamp:
    """Parse a timestamp, interpreting naive values as UTC."""
    ts = pd.Timestamp(timestamp)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
