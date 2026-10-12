"""Synthetic crypto market generator.

Produces a multi-asset perpetual-futures market with the stylised facts that matter for
strategy research, so every module can be tested offline and deterministically:

* fat tails (Student-t innovations) and volatility clustering (GARCH(1,1));
* a common market factor and two persistent macro regimes (calm/bull vs stress/bear);
* a slowly varying latent drift per asset (-> time-series momentum is a real, weak edge);
* 1-bar idiosyncratic reversal (-> short-term reversal is a real, weak edge);
* persistent funding dislocations (leverage demand, partly driven by recent performance) that
  *predict* lower returns (-> funding carry is a real, weak edge);
* volume that co-moves with absolute returns and stress; staggered listings.

Edges are deliberately small: a strategy that looks spectacular on this data is a red flag.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.config.schema import SyntheticConfig
from alphaforge.data.sources.base import DataSource
from alphaforge.utils.time import bars_per_day, periods_per_year, timeframe_to_freq

_BASE_FUNDING_8H = 1e-4  # 0.01% per 8h: the common exchange baseline


@dataclass(frozen=True)
class SyntheticMarket:
    """Output of :func:`generate_synthetic_market`."""

    ohlcv: dict[str, pd.DataFrame]
    funding: dict[str, pd.Series]  # per-bar accrued funding
    basis: dict[str, pd.Series]  # annualised perp premium over spot
    regimes: pd.Series  # 0 = calm, 1 = stress (ground truth, for diagnostics)


def _student_t(rng: np.random.Generator, dof: float, size: int | tuple[int, ...]) -> np.ndarray:
    """Unit-variance Student-t draws."""
    return rng.standard_t(dof, size=size) / np.sqrt(dof / (dof - 2.0))


def generate_synthetic_market(
    symbols: Sequence[str],
    timeframe: str = "1d",
    params: SyntheticConfig | None = None,
    seed: int = 42,
) -> SyntheticMarket:
    """Simulate a correlated crypto perp market. Deterministic for a given ``seed``."""
    p = params or SyntheticConfig()
    rng = np.random.default_rng(seed)
    n, k = p.n_bars, len(symbols)
    ppy = periods_per_year(timeframe)
    per_day = bars_per_day(timeframe)
    sigma_bar = p.annual_vol / np.sqrt(ppy)
    bar_funding_scale = 24.0 / (8.0 * per_day)  # 8h funding periods per bar

    # --- macro regimes: two-state Markov chain with durations in calendar days
    stay = np.array([np.exp(-1.0 / (150 * per_day)), np.exp(-1.0 / (40 * per_day))])
    regimes = np.zeros(n, dtype=int)
    for t in range(1, n):
        prev = regimes[t - 1]
        regimes[t] = prev if rng.random() < stay[prev] else 1 - prev
    regime_vol = np.where(regimes == 1, 1.7, 0.85)
    regime_drift = np.where(regimes == 1, -2.0, 1.0) * p.annual_drift / ppy

    # --- asset heterogeneity
    vol_mult = rng.uniform(0.75, 1.45, k)
    prices0 = 10 ** rng.uniform(-1, 4, k)
    base_dollar_volume = 10 ** rng.uniform(7.0, 9.5, k) / per_day

    # --- innovations
    market = _student_t(rng, p.tail_dof, n)
    idio = _student_t(rng, p.tail_dof, (n, k))
    shocks = np.sqrt(p.factor_share) * market[:, None] + np.sqrt(1 - p.factor_share) * idio

    garch_a, garch_b = 0.08, 0.90
    h = np.ones(k)
    trend = np.zeros(k)
    crowd = np.zeros(k)
    prev_idio = np.zeros(k)
    fund_dev = np.zeros(k)
    fund_rho = np.exp(-1.0 / (10 * per_day))  # ~10-day persistence of leverage demand
    fund_sd = 2.5e-4 * np.sqrt(1 - fund_rho**2)
    phi = p.trend_persistence
    trend_scale = np.sqrt(1 - phi**2) * p.trend_strength * sigma_bar * vol_mult

    log_ret = np.empty((n, k))
    funding_8h = np.empty((n, k))
    sig_path = np.empty((n, k))
    for t in range(n):
        sig = sigma_bar * vol_mult * regime_vol[t] * np.sqrt(h)
        trend = phi * trend + trend_scale * rng.standard_normal(k)
        # funding = baseline + persistent idiosyncratic leverage demand + crowding
        fund_dev = fund_rho * fund_dev + fund_sd * rng.standard_normal(k)
        f8 = _BASE_FUNDING_8H + fund_dev + 0.01 * crowd
        f8 = np.clip(f8, -3e-3, 3e-3)
        carry_penalty = p.carry_strength * (f8 - _BASE_FUNDING_8H) * bar_funding_scale
        idio_part = sig * np.sqrt(1 - p.factor_share) * idio[t]
        mu = regime_drift[t] + trend - carry_penalty - p.reversal_strength * prev_idio
        r = mu + sig * shocks[t] - 0.5 * sig**2
        log_ret[t] = r
        funding_8h[t] = f8
        sig_path[t] = sig
        prev_idio = idio_part
        crowd = 0.9 * crowd + 0.1 * r * per_day
        h = np.clip((1 - garch_a - garch_b) + garch_a * shocks[t] ** 2 + garch_b * h, 0.2, 6.0)

    index = pd.date_range(p.start, periods=n, freq=timeframe_to_freq(timeframe), tz="UTC")
    index.name = "timestamp"
    close = prices0 * np.exp(np.cumsum(log_ret, axis=0))
    prev_close = np.vstack([prices0, close[:-1]])
    open_ = prev_close * np.exp(0.05 * sig_path * rng.standard_normal((n, k)))
    wick = np.abs(rng.standard_normal((2, n, k))) * sig_path * 0.6
    high = np.maximum(open_, close) * np.exp(wick[0])
    low = np.minimum(open_, close) * np.exp(-wick[1])
    activity = np.exp(
        0.8 * np.abs(shocks) + 0.4 * regimes[:, None] + 0.3 * rng.standard_normal((n, k))
    )
    volume = base_dollar_volume * activity / close

    listing = np.zeros(k, dtype=int)
    if p.stagger_listings and k >= 4:
        late = rng.choice(k, size=max(1, k // 4), replace=False)
        listing[late] = rng.integers(n // 10, n // 3, size=len(late))

    ohlcv: dict[str, pd.DataFrame] = {}
    funding: dict[str, pd.Series] = {}
    basis: dict[str, pd.Series] = {}
    for j, sym in enumerate(symbols):
        sl = slice(listing[j], None)
        ohlcv[sym] = pd.DataFrame(
            {
                "open": open_[sl, j],
                "high": high[sl, j],
                "low": low[sl, j],
                "close": close[sl, j],
                "volume": volume[sl, j],
            },
            index=index[sl],
        )
        funding[sym] = pd.Series(funding_8h[sl, j] * bar_funding_scale, index=index[sl])
        noise = 0.02 * rng.standard_normal(n - listing[j])
        basis[sym] = pd.Series(funding_8h[sl, j] * 3 * 365 + noise, index=index[sl])

    return SyntheticMarket(
        ohlcv=ohlcv,
        funding=funding,
        basis=basis,
        regimes=pd.Series(regimes, index=index, name="regime"),
    )


class SyntheticSource(DataSource):
    """:class:`DataSource` facade over :func:`generate_synthetic_market`.

    The whole universe is generated jointly (cross-asset correlation requires it) and cached.
    """

    name = "synthetic"

    def __init__(
        self,
        symbols: Sequence[str],
        timeframe: str = "1d",
        params: SyntheticConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.timeframe = timeframe
        self.market = generate_synthetic_market(symbols, timeframe, params, seed)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if timeframe != self.timeframe:
            raise ValueError(f"Source generated at {self.timeframe}, requested {timeframe}")
        return self.market.ohlcv[symbol].loc[start:end].copy()

    def fetch_funding(
        self,
        symbol: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.Series | None:
        return self.market.funding[symbol].loc[start:end].copy()
