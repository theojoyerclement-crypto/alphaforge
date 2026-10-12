"""Price/volume technical features (trend, volatility, oscillators)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.market_data import MarketData
from alphaforge.features.base import register_feature


@register_feature("returns", "technical", "Log return over the last `lookback` bars")
def log_return(data: MarketData, lookback: int = 1) -> pd.DataFrame:
    return np.log(data.close / data.close.shift(lookback))


@register_feature("momentum", "technical", "Simple return from t-lookback to t-skip")
def momentum(data: MarketData, lookback: int = 20, skip: int = 0) -> pd.DataFrame:
    if skip >= lookback:
        raise ValueError("skip must be smaller than lookback")
    return data.close.shift(skip) / data.close.shift(lookback) - 1.0


@register_feature("vol_adj_momentum", "technical", "Trailing log return scaled by vol (t-stat)")
def vol_adjusted_momentum(
    data: MarketData, lookback: int = 60, vol_lookback: int = 60
) -> pd.DataFrame:
    lr = data.log_returns()
    vol = lr.rolling(vol_lookback, min_periods=max(2, vol_lookback // 2)).std()
    return lr.rolling(lookback).sum() / (vol * np.sqrt(lookback))


@register_feature("realized_vol", "technical", "Annualised close-to-close volatility")
def realized_volatility(data: MarketData, lookback: int = 20) -> pd.DataFrame:
    return data.log_returns().rolling(lookback).std() * np.sqrt(data.periods_per_year)


@register_feature("parkinson_vol", "technical", "Annualised Parkinson (high-low) volatility")
def parkinson_volatility(data: MarketData, lookback: int = 20) -> pd.DataFrame:
    hl = np.log(data.high / data.low) ** 2
    var = hl.rolling(lookback).mean() / (4.0 * np.log(2.0))
    return np.sqrt(var * data.periods_per_year)


@register_feature("garman_klass_vol", "technical", "Annualised Garman-Klass volatility")
def garman_klass_volatility(data: MarketData, lookback: int = 20) -> pd.DataFrame:
    hl = np.log(data.high / data.low) ** 2
    co = np.log(data.close / data.open) ** 2
    var = (0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co).rolling(lookback).mean().clip(lower=0.0)
    return np.sqrt(var * data.periods_per_year)


@register_feature("zscore", "technical", "Z-score of log price vs its trailing mean")
def price_zscore(data: MarketData, lookback: int = 20) -> pd.DataFrame:
    logp = np.log(data.close)
    roll = logp.rolling(lookback)
    return (logp - roll.mean()) / roll.std().replace(0.0, np.nan)


@register_feature("rsi", "technical", "Wilder relative strength index (0-100)")
def rsi(data: MarketData, lookback: int = 14) -> pd.DataFrame:
    delta = data.close.diff()
    up = delta.clip(lower=0.0).ewm(alpha=1.0 / lookback, adjust=False, min_periods=lookback).mean()
    down = (
        (-delta.clip(upper=0.0))
        .ewm(alpha=1.0 / lookback, adjust=False, min_periods=lookback)
        .mean()
    )
    rs = up / down.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


@register_feature("efficiency_ratio", "technical", "Kaufman efficiency ratio (trend quality, 0-1)")
def efficiency_ratio(data: MarketData, lookback: int = 20) -> pd.DataFrame:
    change = (data.close - data.close.shift(lookback)).abs()
    path = data.close.diff().abs().rolling(lookback).sum()
    return change / path.replace(0.0, np.nan)


@register_feature("ema_crossover", "technical", "Fast EMA / slow EMA - 1")
def ema_crossover(data: MarketData, fast: int = 12, slow: int = 48) -> pd.DataFrame:
    if fast >= slow:
        raise ValueError("fast must be smaller than slow")
    ema_f = data.close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_s = data.close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    return ema_f / ema_s - 1.0


@register_feature("distance_to_high", "technical", "Close relative to its trailing maximum")
def distance_to_high(data: MarketData, lookback: int = 60) -> pd.DataFrame:
    return data.close / data.close.rolling(lookback).max() - 1.0


@register_feature("volume_zscore", "technical", "Z-score of log volume vs trailing window")
def volume_zscore(data: MarketData, lookback: int = 20) -> pd.DataFrame:
    logv = np.log(data.volume.replace(0.0, np.nan))
    roll = logv.rolling(lookback, min_periods=max(2, lookback // 2))
    return (logv - roll.mean()) / roll.std().replace(0.0, np.nan)
