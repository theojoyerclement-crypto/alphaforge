from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.alpha import (
    AlphaResearcher,
    benjamini_hochberg,
    cross_sectional_ic,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    forward_returns,
    minimum_track_record_length,
    newey_west_tstat,
    probabilistic_sharpe_ratio,
)
from alphaforge.alpha.base import Alpha
from alphaforge.data.market_data import MarketData


def test_perfect_foresight_has_unit_ic(market: MarketData) -> None:
    fwd = forward_returns(market.close, 1)
    ic = cross_sectional_ic(fwd, fwd)
    assert ic.dropna().min() == pytest.approx(1.0)


def test_noise_alpha_has_zero_ic(market: MarketData) -> None:
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(
        rng.normal(size=market.close.shape), index=market.index, columns=market.symbols
    )
    ic = cross_sectional_ic(noise, forward_returns(market.close, 1))
    assert abs(ic.mean()) < 0.03


def test_newey_west_shrinks_tstat_for_autocorrelated_series() -> None:
    rng = np.random.default_rng(1)
    e = rng.normal(0.05, 1.0, 2000)
    x = pd.Series(np.convolve(e, np.ones(10) / 10, mode="valid"))
    naive = x.mean() / (x.std() / np.sqrt(len(x)))
    assert newey_west_tstat(x, lags=10) < 0.6 * naive


def test_psr_dsr_and_mintrl() -> None:
    rng = np.random.default_rng(2)
    good = pd.Series(rng.normal(0.002, 0.01, 1500))
    psr = probabilistic_sharpe_ratio(good)
    assert psr > 0.99
    # deflation for many trials can only lower confidence
    assert deflated_sharpe_ratio(good, n_trials=100, sr_variance=1 / 1500) < psr
    assert expected_max_sharpe(1, 0.1) == 0.0
    assert minimum_track_record_length(good) < 1500
    assert minimum_track_record_length(-good) == float("inf")


def test_benjamini_hochberg() -> None:
    mask = benjamini_hochberg([0.001, 0.008, 0.04, 0.5, np.nan], fdr=0.05)
    assert mask.tolist() == [True, True, False, False, False]
    assert not benjamini_hochberg([np.nan]).any()


@pytest.mark.slow
def test_researcher_recovers_planted_edges() -> None:
    """The generator plants reversal and carry edges and nothing in low volatility.

    With 12 assets x 2000 bars the researcher must find the planted edges (BH-significant,
    positive IC) and must not certify the absent one.
    """
    from alphaforge.config.schema import DataConfig, SyntheticConfig
    from alphaforge.data.loader import load_market_data

    config = DataConfig(
        symbols=[f"S{i}" for i in range(12)], synthetic=SyntheticConfig(n_bars=2000)
    )
    data, _ = load_market_data(config, seed=42)
    lb = AlphaResearcher(horizons=(1, 5)).evaluate(data).leaderboard
    for planted in ("reversal_1", "basis_carry_5", "funding_carry_21"):
        assert lb.loc[planted, "bh_significant"], planted
        assert lb.loc[planted, "ic_mean"] > 0, planted
    assert not lb.loc["low_vol_60", "bh_significant"]
    assert lb.index[0] in {"reversal_1", "basis_carry_5", "funding_carry_21"}


def test_small_sample_certifies_nothing(market: MarketData) -> None:
    """8 assets x 800 bars is too little data: the multiple-testing guard must hold."""
    lb = AlphaResearcher(horizons=(1, 5)).evaluate(market).leaderboard
    assert {"score", "bh_significant", "deflated_sharpe"} <= set(lb.columns)
    assert lb["bh_significant"].sum() <= 1


def test_combined_alpha_is_causal(market: MarketData) -> None:
    from tests.conftest import perturb_future

    researcher = AlphaResearcher()
    names = ["funding_carry_21", "mom_60"]
    a = researcher.combine(market, names, ic_lookback=60)
    b = researcher.combine(perturb_future(market, 600), names, ic_lookback=60)
    pd.testing.assert_frame_equal(a.iloc[:600], b.iloc[:600])


def test_alpha_masks_untradable(market: MarketData) -> None:
    alpha = Alpha("ones", lambda d: pd.DataFrame(1.0, index=d.index, columns=d.close.columns))
    values = alpha.compute(market)
    assert values.where(market.close.isna()).isna().all().all()
