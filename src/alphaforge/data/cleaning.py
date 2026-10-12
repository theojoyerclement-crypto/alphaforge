"""OHLCV cleaning with an auditable report.

Pipeline (per symbol): normalise to UTC -> sort/de-duplicate -> invalidate non-positive prices
-> repair spike-and-revert bad ticks -> reindex onto the full bar grid -> forward-fill short gaps
(flat bars, zero volume) -> enforce OHLC consistency.

Gaps longer than ``max_ffill_bars`` stay ``NaN``: the asset is treated as untradable there
rather than inventing prices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from alphaforge.config.schema import CleaningConfig
from alphaforge.data.schema import OHLCV_COLUMNS, PRICE_COLUMNS
from alphaforge.utils.time import timeframe_to_freq


@dataclass
class CleaningReport:
    """What the cleaner changed for one symbol."""

    symbol: str
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    invalid_prices: int = 0
    outliers_repaired: int = 0
    gaps_filled: int = 0
    gaps_remaining: int = 0
    ohlc_fixed: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def clean_ohlcv(
    frame: pd.DataFrame,
    timeframe: str,
    config: CleaningConfig | None = None,
    symbol: str = "",
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a single-symbol OHLCV frame.

    Note:
        Outlier repair looks one bar ahead (a bad tick is a spike that immediately reverts).
        That is acceptable for *historical data hygiene*, but it is not something a live
        system could do in real time; keep the threshold high (default 20 MADs).

    Returns:
        ``(clean_frame, report)``.
    """
    cfg = config or CleaningConfig()
    report = CleaningReport(symbol=symbol, rows_in=len(frame))
    out = frame.loc[:, list(OHLCV_COLUMNS)].astype(float).copy()

    idx = pd.DatetimeIndex(out.index)
    out.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out.index.name = "timestamp"
    out = out.sort_index()

    dup = out.index.duplicated(keep="last")
    report.duplicates_removed = int(dup.sum())
    out = out[~dup]

    prices = out[list(PRICE_COLUMNS)]
    invalid = (prices <= 0).any(axis=1)
    report.invalid_prices = int(invalid.sum())
    out.loc[invalid, list(PRICE_COLUMNS)] = np.nan
    out["volume"] = out["volume"].clip(lower=0.0)

    if cfg.repair_outliers and out["close"].notna().sum() > 10:
        log_ret = np.log(out["close"]).diff()
        centered = log_ret - log_ret.median()
        mad = 1.4826 * centered.abs().median()
        if np.isfinite(mad) and mad > 0:
            thr = cfg.outlier_mad_threshold * mad
            nxt = log_ret.shift(-1)
            spike = (log_ret.abs() > thr) & (nxt.abs() > thr) & (np.sign(log_ret) != np.sign(nxt))
            report.outliers_repaired = int(spike.sum())
            out.loc[spike, list(PRICE_COLUMNS)] = np.nan

    valid = out["close"].dropna()
    if valid.empty:
        report.rows_out = 0
        return out.iloc[0:0], report

    grid = pd.date_range(valid.index[0], valid.index[-1], freq=timeframe_to_freq(timeframe))
    grid.name = "timestamp"
    out = out.reindex(grid)

    missing = out["close"].isna()
    filled_close = out["close"].ffill(limit=cfg.max_ffill_bars)
    filled = missing & filled_close.notna()
    for col in PRICE_COLUMNS:
        out[col] = out[col].where(~filled, filled_close)
    out["volume"] = out["volume"].where(~filled, 0.0)
    report.gaps_filled = int(filled.sum())
    report.gaps_remaining = int(out["close"].isna().sum())

    has_price = out["close"].notna()
    hi = out[list(PRICE_COLUMNS)].max(axis=1)
    lo = out[list(PRICE_COLUMNS)].min(axis=1)
    fix = has_price & (
        (out["high"] < hi) | (out["low"] > lo) | out["high"].isna() | out["low"].isna()
    )
    report.ohlc_fixed = int(fix.sum())
    out["high"] = out["high"].where(~has_price, hi)
    out["low"] = out["low"].where(~has_price, lo)
    out["open"] = out["open"].where(~(has_price & out["open"].isna()), out["close"].shift(1))
    out["open"] = out["open"].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0.0)

    report.rows_out = len(out)
    return out, report


def aggregate_funding_to_bars(
    events: pd.Series, index: pd.DatetimeIndex, timeframe: str
) -> pd.Series:
    """Sum discrete funding settlements into the bars they fall in.

    Bars are labelled by their open time, so a settlement at time ``s`` belongs to the bar
    ``floor(s, timeframe)``. Bars without settlements accrue zero funding.
    """
    if events.empty:
        return pd.Series(0.0, index=index)
    ev = events.copy()
    ev_index = pd.DatetimeIndex(ev.index)
    ev.index = ev_index.tz_localize("UTC") if ev_index.tz is None else ev_index.tz_convert("UTC")
    binned = ev.groupby(ev.index.floor(timeframe_to_freq(timeframe))).sum()
    return binned.reindex(index, fill_value=0.0).astype(float)
