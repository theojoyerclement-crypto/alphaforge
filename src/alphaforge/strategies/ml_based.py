"""ML-driven strategies. All set ``requires_fit = True`` and must run under walk-forward."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from alphaforge.data.market_data import MarketData
from alphaforge.features.transforms import cs_demean
from alphaforge.ml.meta_labeling import MetaLabeler
from alphaforge.ml.models import DirectionalModel
from alphaforge.ml.regime import GaussianMixtureRegimeModel, market_regime_features
from alphaforge.strategies.base import Strategy, build_strategy, register_strategy
from alphaforge.utils.logging import get_logger

logger = get_logger(__name__)


@register_strategy
class MLDirectional(Strategy):
    """Trade the probability output of a pooled gradient-boosting direction model.

    ``signal = clip(4 * (p - 0.5), -1, 1)``, zeroed when ``|p - 0.5| < min_edge``, and
    cross-sectionally demeaned when ``market_neutral``. Probabilities are EWMA-smoothed
    (``smoothing_halflife`` bars, causal) because raw tree-model outputs are noisy bar to bar
    and turnover, not accuracy, is usually what kills ML strategies after costs.
    """

    name = "ml_directional"
    family = "machine_learning"
    requires_fit = True

    def __init__(
        self,
        model: str = "lightgbm",
        horizon: int = 5,
        min_edge: float = 0.02,
        market_neutral: bool = True,
        smoothing_halflife: float | None = 3.0,
        random_state: int = 42,
        model_params: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.horizon = horizon
        self.min_edge = min_edge
        self.market_neutral = market_neutral
        self.smoothing_halflife = smoothing_halflife
        self.random_state = random_state
        self.model_params = dict(model_params or {})
        self._model: DirectionalModel | None = None

    def fit(self, data: MarketData) -> MLDirectional:
        self._model = DirectionalModel(
            self.model,
            self.horizon,
            relative=self.market_neutral,
            random_state=self.random_state,
            model_params=self.model_params,
        ).fit(data)
        return self

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("MLDirectional must be fitted first")
        proba = self._model.predict_proba(data)
        if self.smoothing_halflife:
            proba = proba.ewm(halflife=self.smoothing_halflife, ignore_na=True).mean()
        edge = proba - 0.5
        signal = (4.0 * edge).clip(-1.0, 1.0).where(edge.abs() >= self.min_edge, 0.0)
        if self.market_neutral:
            signal = cs_demean(signal)
        return self.finalize(signal, data)


@register_strategy
class MetaLabeledStrategy(Strategy):
    """A rule-based primary strategy filtered and sized by a meta-labeling classifier."""

    name = "meta_labeled"
    family = "machine_learning"
    requires_fit = True

    def __init__(
        self,
        primary: str = "ts_momentum",
        primary_params: Mapping[str, Any] | None = None,
        model: str = "random_forest",
        pt_mult: float = 1.5,
        sl_mult: float = 1.0,
        max_holding: int = 10,
        min_probability: float = 0.52,
        random_state: int = 42,
    ) -> None:
        self.primary = primary
        self.primary_params = dict(primary_params or {})
        self.model = model
        self.pt_mult = pt_mult
        self.sl_mult = sl_mult
        self.max_holding = max_holding
        self.min_probability = min_probability
        self.random_state = random_state
        self._primary = build_strategy(primary, self.primary_params)
        self._meta: MetaLabeler | None = None

    def fit(self, data: MarketData) -> MetaLabeledStrategy:
        primary_signals = self._primary.fit(data).generate_signals(data)
        self._meta = MetaLabeler(
            self.model,
            self.pt_mult,
            self.sl_mult,
            self.max_holding,
            min_probability=self.min_probability,
            random_state=self.random_state,
        ).fit(data, primary_signals)
        return self

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        if self._meta is None:
            raise RuntimeError("MetaLabeledStrategy must be fitted first")
        primary_signals = self._primary.generate_signals(data)
        return self.finalize(self._meta.size_positions(data, primary_signals), data)


@register_strategy
class RegimeAdaptive(Strategy):
    """Learn which candidate strategy works in which market regime.

    On the training window: fit a Gaussian-mixture regime model on market state features,
    then for every regime pick the candidate with the best in-sample Sharpe (flat if none is
    positive). Out of sample: detect the regime bar by bar and run the chosen candidate.
    """

    name = "ml_regime"
    family = "regime"
    requires_fit = True

    def __init__(
        self,
        candidates: Mapping[str, Mapping[str, Any]] | None = None,
        n_regimes: int = 3,
        min_regime_bars: int = 30,
        random_state: int = 42,
    ) -> None:
        self.candidates = {
            k: dict(v)
            for k, v in (
                candidates
                or {
                    "ts_momentum": {},
                    "zscore_reversion": {},
                    "funding_carry": {},
                }
            ).items()
        }
        self.n_regimes = n_regimes
        self.min_regime_bars = min_regime_bars
        self.random_state = random_state
        self._regime_model: GaussianMixtureRegimeModel | None = None
        self.selection_: dict[int, str] = {}

    def _candidate_signals(self, data: MarketData) -> dict[str, pd.DataFrame]:
        out = {}
        for name, params in self.candidates.items():
            try:
                out[name] = build_strategy(name, params).generate_signals(data)
            except ValueError as exc:  # e.g. carry without funding data
                logger.warning("Skipping candidate %s: %s", name, exc)
        return out

    def fit(self, data: MarketData) -> RegimeAdaptive:
        features = market_regime_features(data)
        self._regime_model = GaussianMixtureRegimeModel(self.n_regimes, self.random_state).fit(
            features
        )
        regimes = self._regime_model.predict(features)
        rets = data.returns().fillna(0.0)
        best: dict[int, tuple[str, float]] = {}
        for name, sig in self._candidate_signals(data).items():
            pnl = (sig.shift(1) * rets).sum(axis=1)
            for regime in range(self.n_regimes):
                # pnl at t comes from positions decided in the regime observed at t-1
                in_regime = pnl[regimes.shift(1) == regime]
                if len(in_regime) < self.min_regime_bars or in_regime.std() == 0:
                    continue
                sharpe = float(in_regime.mean() / in_regime.std())
                if sharpe > best.get(regime, ("", 0.0))[1]:
                    best[regime] = (name, sharpe)
        self.selection_ = {regime: name for regime, (name, _) in best.items()}
        logger.info("Regime -> strategy map: %s", self.selection_)
        return self

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        if self._regime_model is None:
            raise RuntimeError("RegimeAdaptive must be fitted first")
        regimes = self._regime_model.predict(market_regime_features(data))
        signals = self._candidate_signals(data)
        out = pd.DataFrame(0.0, index=data.close.index, columns=data.close.columns)
        for regime, name in self.selection_.items():
            if name not in signals:
                continue
            mask = (regimes == regime).to_numpy()
            out.loc[mask] = signals[name].loc[mask].to_numpy()
        return self.finalize(out, data)
