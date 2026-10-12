"""Alpha research workflow: evaluate a library, rank it honestly, combine the survivors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.alpha.base import Alpha
from alphaforge.alpha.evaluation import (
    AlphaEvaluation,
    cross_sectional_ic,
    evaluate_alpha,
    forward_returns,
)
from alphaforge.alpha.library import default_alpha_library
from alphaforge.alpha.validation import benjamini_hochberg, deflated_sharpe_ratio
from alphaforge.data.market_data import MarketData
from alphaforge.features.transforms import cs_zscore
from alphaforge.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AlphaResearchReport:
    """Results of an alpha research pass."""

    leaderboard: pd.DataFrame
    evaluations: dict[str, AlphaEvaluation]
    ic_series: pd.DataFrame
    long_short_returns: pd.DataFrame

    def ic_decay(self) -> pd.DataFrame:
        """Mean IC by forward horizon (rows: alphas, columns: horizons)."""
        return pd.DataFrame({k: v.ic_decay for k, v in self.evaluations.items()}).T


class AlphaResearcher:
    """Evaluate, validate and rank a set of alphas.

    The score is deliberately conservative:

    ``score = |annualised ICIR| x (1 + rank autocorrelation) / 2 x DSR``, where the DSR is the
    deflated Sharpe of the alpha's (direction-aligned) long-short portfolio given the number
    of trials (each alpha counts twice: it could have been traded either way). Alphas failing
    Benjamini-Hochberg at the configured FDR are flagged and their score halved.

    Args:
        alphas: Candidates (defaults to :func:`default_alpha_library`).
        horizons: Forward horizons for the IC decay profile; the first is the primary one.
        n_quantiles: Buckets for quantile spread analysis.
        fdr: False discovery rate for Benjamini-Hochberg.
    """

    def __init__(
        self,
        alphas: Sequence[Alpha] | None = None,
        horizons: Sequence[int] = (1, 5, 10, 20),
        n_quantiles: int = 3,
        fdr: float = 0.10,
    ) -> None:
        self.alphas = list(alphas) if alphas is not None else default_alpha_library()
        self.horizons = sorted(horizons)
        self.n_quantiles = n_quantiles
        self.fdr = fdr

    def evaluate(self, data: MarketData) -> AlphaResearchReport:
        """Evaluate every alpha on ``data`` and build the leaderboard."""
        evaluations: dict[str, AlphaEvaluation] = {}
        ics: dict[str, pd.Series] = {}
        ls: dict[str, pd.Series] = {}
        for alpha in self.alphas:
            values = alpha.compute(data)
            if values.notna().sum().sum() == 0:
                logger.info("Skipping %s: no data (missing inputs?)", alpha.name)
                continue
            ev, ic, ls_ret = evaluate_alpha(
                alpha.name, values, data, self.horizons, self.n_quantiles
            )
            evaluations[alpha.name] = ev
            ics[alpha.name] = ic
            ls[alpha.name] = ls_ret

        ls_frame = pd.DataFrame(ls)
        names = list(evaluations)
        bh = benjamini_hochberg([evaluations[n].ic_pvalue for n in names], self.fdr)
        # Each alpha is tried in both directions -> 2N trials. The Sharpe variance under the
        # null of no skill is ~1/T per period; using the cross-alpha dispersion instead would
        # punish genuinely different signals for being different.
        n_trials = 2 * len(names)
        sr_null_var = 1.0 / max(len(ls_frame.dropna(how="all")), 1)

        rows = []
        for name, significant in zip(names, bh, strict=True):
            ev = evaluations[name]
            direction = 1.0 if ev.ic_mean >= 0 else -1.0
            dsr = deflated_sharpe_ratio(direction * ls_frame[name], n_trials, sr_null_var)
            turnover_penalty = (1.0 + ev.rank_autocorr) / 2.0
            score = abs(ev.icir_annualized) * turnover_penalty * (dsr if np.isfinite(dsr) else 0.0)
            score *= 1.0 if significant else 0.5
            category = next(a.category for a in self.alphas if a.name == name)
            rows.append(
                {
                    "alpha": name,
                    "category": category,
                    "direction": "long" if direction > 0 else "inverted",
                    "ic_mean": ev.ic_mean,
                    "icir_ann": ev.icir_annualized,
                    "ic_tstat_nw": ev.ic_tstat,
                    "ic_pvalue": ev.ic_pvalue,
                    "bh_significant": bool(significant),
                    "hit_rate": ev.ic_hit_rate,
                    "rank_autocorr": ev.rank_autocorr,
                    "ls_sharpe": ev.long_short_sharpe,
                    "deflated_sharpe": dsr,
                    "quantile_spread": ev.quantile_spread,
                    "score": score,
                }
            )
        leaderboard = pd.DataFrame(rows).set_index("alpha").sort_values("score", ascending=False)
        return AlphaResearchReport(leaderboard, evaluations, pd.DataFrame(ics), ls_frame)

    def combine(
        self,
        data: MarketData,
        alpha_names: Sequence[str],
        ic_lookback: int = 120,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Causal IC-weighted composite of several alphas.

        Each alpha is cross-sectionally z-scored and weighted by its trailing mean IC. The IC
        at ``t`` needs returns up to ``t + horizon``, so ICs are lagged by ``horizon + 1``
        bars before use: the composite never sees an IC that was not yet realised.
        """
        by_name = {a.name: a for a in self.alphas}
        fwd = forward_returns(data.close, horizon)
        num = pd.DataFrame(0.0, index=data.close.index, columns=data.close.columns)
        den = pd.Series(0.0, index=data.close.index)
        for name in alpha_names:
            values = by_name[name].compute(data)
            z = cs_zscore(values).fillna(0.0)
            ic = cross_sectional_ic(values, fwd).shift(horizon + 1)
            w = ic.rolling(ic_lookback, min_periods=ic_lookback // 3).mean().fillna(0.0)
            num = num + z.mul(w, axis=0)
            den = den + w.abs()
        composite = num.div(den.replace(0.0, np.nan), axis=0)
        return composite.where(data.close.notna())
