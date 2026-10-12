"""``MarketData``: the aligned multi-asset panel every downstream layer consumes.

Conventions
-----------
* Each field is a *wide* DataFrame: index = UTC bar timestamps, columns = symbols.
* A bar's values are known at the bar's close. A signal computed from row ``t`` may only be
  traded at the close of ``t`` or later (see ``BacktestConfig.execution_delay``).
* ``NaN`` close means the asset is not tradable on that bar (not listed yet, or a data gap).
* ``funding`` is the funding rate *accrued over the bar*, as a fraction of notional.
  Positive funding means longs pay shorts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.data.schema import OHLCV_COLUMNS
from alphaforge.utils.time import periods_per_year


@dataclass(frozen=True)
class MarketData:
    """Aligned OHLCV (+ optional funding and extra) panels for a universe of symbols."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    timeframe: str = "1d"
    funding: pd.DataFrame | None = None
    extras: Mapping[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.close.index, pd.DatetimeIndex):
            raise TypeError("MarketData panels must be indexed by a DatetimeIndex")
        if not self.close.index.is_monotonic_increasing or self.close.index.has_duplicates:
            raise ValueError("MarketData index must be strictly increasing")
        for name, panel in self.panels():
            if not panel.index.equals(self.close.index):
                raise ValueError(f"Panel {name!r} index is not aligned with close")
            if not panel.columns.equals(self.close.columns):
                raise ValueError(f"Panel {name!r} columns are not aligned with close")

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_frames(
        cls,
        frames: Mapping[str, pd.DataFrame],
        timeframe: str,
        funding: Mapping[str, pd.Series] | None = None,
        extras: Mapping[str, Mapping[str, pd.Series]] | None = None,
    ) -> MarketData:
        """Build a panel from per-symbol OHLCV frames (outer-joined on time).

        Args:
            frames: ``{symbol: DataFrame[open, high, low, close, volume]}`` with UTC index.
            timeframe: Bar timeframe (``"1h"``, ``"1d"``...).
            funding: Optional ``{symbol: Series}`` of per-bar accrued funding.
            extras: Optional ``{field: {symbol: Series}}`` of additional panels (e.g. basis).
        """
        if not frames:
            raise ValueError("At least one symbol frame is required")
        symbols = list(frames)
        index = frames[symbols[0]].index
        for sym in symbols[1:]:
            index = index.union(frames[sym].index)
        index = pd.DatetimeIndex(index).sort_values()
        index.name = "timestamp"

        def panel(series_by_symbol: Mapping[str, pd.Series]) -> pd.DataFrame:
            out = pd.DataFrame(
                {s: series_by_symbol[s].reindex(index) for s in symbols if s in series_by_symbol},
                index=index,
                columns=pd.Index(symbols, name="symbol"),
            )
            return out.astype(float)

        ohlcv = {col: panel({s: frames[s][col] for s in symbols}) for col in OHLCV_COLUMNS}

        funding_panel = None
        if funding:
            funding_panel = panel(dict(funding)).fillna(0.0).where(ohlcv["close"].notna())

        extra_panels = {name: panel(dict(by_sym)) for name, by_sym in (extras or {}).items()}

        return cls(
            open=ohlcv["open"],
            high=ohlcv["high"],
            low=ohlcv["low"],
            close=ohlcv["close"],
            volume=ohlcv["volume"],
            timeframe=timeframe,
            funding=funding_panel,
            extras=extra_panels,
        )

    # ------------------------------------------------------------------ accessors
    def panels(self) -> Iterator[tuple[str, pd.DataFrame]]:
        """Iterate over every non-null panel as ``(name, frame)``."""
        yield "open", self.open
        yield "high", self.high
        yield "low", self.low
        yield "close", self.close
        yield "volume", self.volume
        if self.funding is not None:
            yield "funding", self.funding
        yield from self.extras.items()

    @property
    def symbols(self) -> list[str]:
        return [str(c) for c in self.close.columns]

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    @property
    def n_bars(self) -> int:
        return len(self.close)

    @property
    def periods_per_year(self) -> float:
        return periods_per_year(self.timeframe)

    @property
    def has_funding(self) -> bool:
        return self.funding is not None and bool(self.funding.abs().sum().sum() > 0)

    def returns(self) -> pd.DataFrame:
        """Simple close-to-close returns; ``NaN`` where either close is missing."""
        return self.close / self.close.shift(1) - 1.0

    def log_returns(self) -> pd.DataFrame:
        """Close-to-close log returns."""
        return np.log(self.close / self.close.shift(1))

    def dollar_volume(self) -> pd.DataFrame:
        """Traded notional per bar (close x base volume)."""
        return self.close * self.volume

    # ------------------------------------------------------------------ slicing
    def _map(self, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> MarketData:
        return MarketData(
            open=fn(self.open),
            high=fn(self.high),
            low=fn(self.low),
            close=fn(self.close),
            volume=fn(self.volume),
            timeframe=self.timeframe,
            funding=None if self.funding is None else fn(self.funding),
            extras={k: fn(v) for k, v in self.extras.items()},
        )

    def iloc(self, start: int | None = None, stop: int | None = None) -> MarketData:
        """Positional row slice ``[start, stop)``."""
        return self._map(lambda f: f.iloc[start:stop])

    def loc(
        self, start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None
    ) -> MarketData:
        """Label-based (inclusive) time slice."""
        return self._map(lambda f: f.loc[start:end])

    def select(self, symbols: list[str]) -> MarketData:
        """Restrict to a subset of symbols."""
        return self._map(lambda f: f.loc[:, symbols])
