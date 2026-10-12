"""Market regime detection: a causal rule-based classifier and a Gaussian mixture model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from alphaforge.data.market_data import MarketData


def market_regime_features(
    data: MarketData, vol_lookback: int = 20, trend_lookback: int = 60
) -> pd.DataFrame:
    """Market-level state variables built from the equal-weight universe index.

    Columns: ``volatility`` (annualised), ``trend`` (t-stat of trailing return),
    ``dispersion`` (cross-sectional return dispersion), ``drawdown`` (from trailing peak).
    """
    rets = data.returns()
    market = rets.mean(axis=1)
    ppy = data.periods_per_year
    vol = market.rolling(vol_lookback).std()
    trend = market.rolling(trend_lookback).sum() / (vol * np.sqrt(trend_lookback))
    level = (1.0 + market.fillna(0.0)).cumprod()
    return pd.DataFrame(
        {
            "volatility": vol * np.sqrt(ppy),
            "trend": trend,
            "dispersion": rets.std(axis=1).rolling(vol_lookback).mean(),
            "drawdown": level / level.rolling(trend_lookback, min_periods=1).max() - 1.0,
        }
    )


class VolatilityRegimeClassifier:
    """Rule-based volatility regimes from *trailing* quantiles (no fitting, fully causal).

    Labels: 0 = low, 1 = normal, 2 = high volatility.
    """

    def __init__(self, window: int = 365, low_q: float = 0.33, high_q: float = 0.66) -> None:
        self.window = window
        self.low_q = low_q
        self.high_q = high_q

    def predict(self, volatility: pd.Series) -> pd.Series:
        roll = volatility.rolling(self.window, min_periods=self.window // 3)
        lo, hi = roll.quantile(self.low_q), roll.quantile(self.high_q)
        regime = pd.Series(1.0, index=volatility.index)
        regime[volatility <= lo] = 0.0
        regime[volatility >= hi] = 2.0
        return regime.where(lo.notna() & volatility.notna()).rename("regime")


class GaussianMixtureRegimeModel:
    """Unsupervised regimes from a Gaussian mixture on standardised state features.

    States are relabelled by ascending mean of the first feature (volatility by default) so
    that regime ``0`` is always the calmest, which keeps labels stable across refits.
    Prediction at ``t`` uses only the features at ``t``; fit only on training windows.
    """

    def __init__(
        self, n_regimes: int = 3, random_state: int = 42, covariance_type: str = "full"
    ) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.scaler_: StandardScaler | None = None
        self.model_: GaussianMixture | None = None
        self.order_: np.ndarray | None = None
        self.columns_: list[str] | None = None

    def fit(self, features: pd.DataFrame) -> GaussianMixtureRegimeModel:
        clean = features.dropna()
        if len(clean) < 10 * self.n_regimes:
            raise ValueError("Not enough observations to fit the regime model")
        self.columns_ = list(features.columns)
        self.scaler_ = StandardScaler().fit(clean.to_numpy())
        self.model_ = GaussianMixture(
            self.n_regimes,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=5,
        ).fit(self.scaler_.transform(clean.to_numpy()))
        self.order_ = np.argsort(np.argsort(self.model_.means_[:, 0]))
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None or self.scaler_ is None or self.order_ is None:
            raise RuntimeError("Regime model is not fitted")
        X = features[self.columns_]
        valid = X.notna().all(axis=1)
        out = pd.DataFrame(np.nan, index=X.index, columns=range(self.n_regimes))
        if valid.any():
            raw = self.model_.predict_proba(self.scaler_.transform(X[valid].to_numpy()))
            ordered = np.empty_like(raw)
            ordered[:, self.order_] = raw
            out.loc[valid] = ordered
        return out

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Most likely regime per bar (``NaN`` where features are incomplete)."""
        proba = self.predict_proba(features)
        labels = proba.to_numpy().argmax(axis=1).astype(float)
        labels[proba.isna().any(axis=1).to_numpy()] = np.nan
        return pd.Series(labels, index=features.index, name="regime")
