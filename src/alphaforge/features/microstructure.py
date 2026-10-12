"""Microstructure proxies estimated from OHLCV bars.

Without order-book data, liquidity and order-flow can still be approximated from bars:

* Amihud (2002) illiquidity: price impact per dollar traded.
* Roll (1984) effective spread from negative serial covariance of price changes.
* Corwin & Schultz (2012) spread from two-bar high/low ranges.
* Kyle's lambda proxy: slope of returns on signed dollar volume.
* Close location value: where the bar closed in its range, a buy/sell pressure proxy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.market_data import MarketData
from alphaforge.features.base import register_feature


@register_feature("amihud", "microstructure", "Amihud illiquidity: |return| per $1M traded")
def amihud_illiquidity(data: MarketData, lookback: int = 20) -> pd.DataFrame:
    dollar = data.dollar_volume().replace(0.0, np.nan)
    ratio = data.returns().abs() / dollar * 1e6
    return ratio.rolling(lookback, min_periods=max(2, lookback // 2)).mean()


@register_feature("roll_spread", "microstructure", "Roll implied spread (fraction of price)")
def roll_spread(data: MarketData, lookback: int = 20) -> pd.DataFrame:
    dp = data.close.diff()
    cov = dp.rolling(lookback).cov(dp.shift(1), pairwise=False)
    return 2.0 * np.sqrt((-cov).clip(lower=0.0)) / data.close


@register_feature("corwin_schultz", "microstructure", "Corwin-Schultz high-low spread estimate")
def corwin_schultz_spread(data: MarketData, lookback: int = 20) -> pd.DataFrame:
    log_hl = np.log(data.high / data.low)
    beta = log_hl**2 + log_hl.shift(1) ** 2
    high2 = np.maximum(data.high, data.high.shift(1))
    low2 = np.minimum(data.low, data.low.shift(1))
    gamma = np.log(high2 / low2) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = (2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))).clip(lower=0.0)
    return spread.rolling(lookback, min_periods=max(2, lookback // 2)).mean()


@register_feature("kyle_lambda", "microstructure", "Return sensitivity to signed $ volume (x1e9)")
def kyle_lambda(data: MarketData, lookback: int = 20) -> pd.DataFrame:
    ret = data.returns()
    signed = np.sign(ret) * data.dollar_volume()
    cov = ret.rolling(lookback).cov(signed, pairwise=False)
    var = signed.rolling(lookback).var().replace(0.0, np.nan)
    return cov / var * 1e9


@register_feature("close_location", "microstructure", "Volume-weighted close location value")
def close_location(data: MarketData, lookback: int = 10) -> pd.DataFrame:
    rng = (data.high - data.low).replace(0.0, np.nan)
    clv = ((data.close - data.low) - (data.high - data.close)) / rng
    clv = clv.fillna(0.0).where(data.close.notna())
    num = (clv * data.volume).rolling(lookback).sum()
    den = data.volume.rolling(lookback).sum().replace(0.0, np.nan)
    return num / den


@register_feature("hl_range", "microstructure", "Mean log high-low range")
def high_low_range(data: MarketData, lookback: int = 5) -> pd.DataFrame:
    return np.log(data.high / data.low).rolling(lookback).mean()
