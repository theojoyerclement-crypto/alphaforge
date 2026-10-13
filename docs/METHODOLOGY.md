# Research methodology

## Design choices that matter to a hiring manager

* **Causality.** A feature at bar *t* may use data ≤ *t* and nothing else. Forward returns
  appear in only two places, as *targets* (`alpha/evaluation.py`, `ml/labeling.py`).
  Property tests scramble the future and require the past to be bit-identical. The paper
  trader re-derives every decision from truncated history, and the result must match the
  backtest exactly.
* **Frictions.** The defaults are 1 bar of latency, 4 bps fee, 1.5 bps half-spread,
  square-root impact on ADV and volatility, and bar-by-bar perp funding. The cost-stress
  config shows a t = 5.5 alpha going from Sharpe +1.8 to −2.9 as these are switched on. On
  real Binance data, costs consume about half of the gross edge (§5).
* **Multiple testing.** Fifteen alphas tested in two directions make 30 trials. ICs get
  Newey-West t-stats, the library is filtered by Benjamini-Hochberg FDR, and long-short
  Sharpes are deflated against the expected maximum of 30 skill-less trials. On too small a
  sample the stack certifies nothing, and a test checks that too.
* **Reproducibility.** One YAML file plus the data fully determine a run. The manifest stores
  the config hash, data hash, git commit, a dirty-tree flag, the seed and package versions.
  The parquet store keeps a content hash per file. Real-data tables pin `data.end`, so they
  can be re-run exactly.
* **Reporting discipline.** Results are published as they come out, including ML that loses
  to a carry rule and real-data results that do not beat buy & hold on a risk-adjusted basis.

## 1. From idea to strategy

1. **Hypothesis.** Write down the economic rationale *before* looking at results
   (each library alpha carries one in its `description`).
2. **Evidence on the signal.** Rank IC per bar, its mean, ICIR, IC decay across horizons,
   quantile spreads and rank autocorrelation (a turnover proxy). Evaluated on cross-sectional
   ranks so scale and outliers do not matter.
3. **Significance under dependence.** Forward returns over *h* bars overlap, so consecutive ICs
   are autocorrelated. t-statistics use a Newey-West (HAC) standard error.
4. **Significance under multiplicity.** Testing *N* alphas guarantees lucky ones.
   Benjamini-Hochberg controls the false discovery rate across the library, and the Deflated
   Sharpe Ratio benchmarks each long-short Sharpe against the expected *maximum* Sharpe of
   *2N* skill-less trials (each alpha could have been traded in either direction).
5. **Tradability.** A significant IC says nothing about costs. The strategy is backtested with
   execution delay, fees, spread, square-root impact and funding, and its PnL is attributed to
   price, funding and costs.
6. **Out-of-sample discipline.** Anything that learns from data runs under walk-forward:
   fit on the past, trade the next unseen window, refit, repeat.

## 2. Bias checklist

| Bias | Control | Where |
|---|---|---|
| Look-ahead in features/signals | Future-scrambling property tests | `tests/test_features.py`, `tests/test_strategies.py` |
| Look-ahead in composites | IC weights lagged by `horizon + 1` | `AlphaResearcher.combine` |
| Look-ahead in risk models | Estimates at *t* use data ≤ *t* | `portfolio.sizing`, `StrategyAllocator` |
| Same-bar execution | `execution_delay ≥ 1` default | `BacktestEngine` |
| Label leakage in CV | Purging on `[t0, t1]` overlap + embargo | `backtest.cv` |
| Incomplete labels at the data edge | Triple-barrier events without a known outcome are dropped | `ml.labeling` |
| Model state leaking across folds | Fresh strategy instances per fold | `WalkForwardRunner` |
| In-sample ML evaluation | `ConfigurationError` without walk-forward | `research.experiment` |
| Listing bias | NaN before listing = untradable; never forward-filled | `data.cleaning`, `MarketData` |
| Unfinished bars | The exchange's still-forming candle is dropped at ingestion | `CCXTSource.fetch_ohlcv` |
| Dead feeds (placeholder candles after delisting/halts) | Runs of flat, zero-volume bars set untradable | `clean_ohlcv(stale_bars)` |
| Selection hindsight in the universe | Point-in-time membership: listing age + trailing liquidity rank | `data.universe`, `universe_mask` |
| Silent data defects | Per-symbol audit (gaps, dead feeds, funding coverage, early end); `on_fail: error` stops the run | `data.quality`, `alphaforge validate` |
| Data snooping | Benjamini-Hochberg + Deflated Sharpe | `alpha.validation` |
| Ignoring carry | Funding charged bar by bar on every position | `BacktestEngine` |

Known residual limitations, stated plainly:

* Outlier repair looks one bar ahead (a bad tick is a spike that reverts). This is acceptable as
  historical hygiene, but a live system could not do it. The threshold is deliberately high (20 MADs).
* `binance_perps.yaml` keeps a fixed, hindsight-picked universe as a baseline. The
  point-in-time variant removes the *selection* hindsight but not survivorship in the data:
  instruments that the venue no longer serves at all (LUNA, MATIC) cannot be candidates.
* A delisted position is flattened at the last real close, at normal costs. Real delistings
  often gap down first, so this is optimistic.
* Impact parameters are literature defaults (`k = 0.1`), not calibrated on fills.
* The drawdown controller caps the loss *per episode*: after a hard stop and cooldown, the
  high-water mark resets. Cumulative drawdown across episodes can exceed `max_drawdown`.

## 3. Reading the ML results correctly

* **AUC slightly below 0.5 under K-fold means "no edge", not "invert the model".** When the true
  signal is null, the fit on the other folds is anti-correlated with the held-out fold, which
  biases cross-validated accuracy below chance.
* **Turnover kills ML strategies more often than accuracy does.** Raw tree outputs jitter bar to
  bar; `MLDirectional` smooths probabilities (EWMA) and ignores small edges (`min_edge`).
* **Meta-labeling trades recall for precision.** Expect fewer trades, lower volatility and lower
  drawdown; a higher Sharpe is possible but not guaranteed.

## 4. The synthetic market as a test oracle

`data.sources.synthetic` plants three weak edges (latent-drift trend, 1-bar idiosyncratic
reversal, funding-predicts-returns carry) and none in low volatility, liquidity or close-location.
`tests/test_alpha.py::test_researcher_recovers_planted_edges` requires the research stack to
certify the planted edges and reject the absent one; `test_small_sample_certifies_nothing`
requires it to stay silent when the sample is too small. Edges are calibrated to be modest
(long-short Sharpe 0.5–2 before costs), so a spectacular result on this data is itself a red flag.

## 5. What changed on real data

`configs/binance_perps.yaml` was written before the real data was looked at and has not been
tuned since. Running it on Binance USDT-M perps (Jul 2021 – Oct 2026) gives a net Sharpe of
0.52 with overlay v2 (0.35 under v1), against 0.44 for BTC buy & hold. The drawdown is −33%
instead of −78%, with near-zero beta. None of these real-data Sharpes is distinguishable
from zero or from BTC at 95% (§7). Three
lessons generalise:

* **Hindsight universes flatter results.** Picking the 12 coins that are large *today* is a
  selection on the outcome. The point-in-time variant (`binance_perps_pit.yaml`) trades the
  monthly top 12 by trailing liquidity among 29 candidates, one of them delisted. Overlay off,
  the portfolio Sharpe falls from +0.53 to −0.15, and funding carry from +0.38 to −0.32.
  Only time-series momentum holds up.
* **Synthetic edges were stronger than reality.** Funding carry has Sharpe 2.0 on the
  synthetic market, and at best 0.38 on real data.
* **A rank IC is not a PnL.** `low_vol_60` has the largest rank IC (t = 8) and a *negative*
  long-short Sharpe. An IC weights every day equally, while dollar PnL is dominated by the few
  high-dispersion days when volatile alts rally.

Two failure modes surfaced only on real data and are now handled in code:

* **Dead feeds after delisting.** Binance keeps serving flat, zero-volume candles for a
  delisted perp (FTM: 644 days of them). Treated as prices, they give the sizer a
  zero-volatility asset and a maximal position. `clean_ohlcv` drops runs of `stale_bars` flat,
  zero-volume bars, and `alphaforge validate` reports them.
* **Path-dependent overlays can trap a book.** With a linear de-risking band between the soft
  and hard drawdown limits, the v1 overlay kept the PIT book at 5–10% exposure for four years.
  The pre-registered fix (v2) and its outcome are in §6.

Full tables and caveats are in [RESULTS.md](RESULTS.md#5-real-market-binance-usdt-m-perpetuals).

## 6. Pre-registered change: drawdown overlay v2 (2026-10-13)

Written and committed **before** any backtest was re-run with the new rule. The commit that
adds this section precedes the commit with the results, so the order can be checked in git.

**Diagnosis.** The v1 controller measures drawdown on the equity *it produces itself*. Inside
the de-risking band, exposure falls, so equity stops moving, so drawdown stops changing. The
book can neither recover its high-water mark nor reach the hard stop that would reset it. In
`binance_perps_pit` it sat at 5–10% exposure for four years. The thresholds are not the
problem; the feedback loop is.

**Rule v2.** The drawdown is measured on a **shadow book**: the same targets at full
exposure, with the same execution delay, costs and funding, simulated bar by bar alongside the
real book. It is exactly the "overlay off" book. The exposure multiplier is the v1 function
applied to the shadow drawdown:

* shadow drawdown < `drawdown_soft_limit`: multiplier 1;
* between the soft limit and `max_drawdown`: falls linearly from 1 to 0;
* at or beyond `max_drawdown`: 0 for `cooldown_bars`, then the shadow high-water mark resets
  to the current shadow equity.

The overlay therefore de-risks while *the strategy* is in drawdown and re-risks when *the
strategy* recovers, whatever the overlay itself did. Everything stays causal: the shadow NAV at
bar *t* uses data ≤ *t*.

**Fixed in advance:**

* The parameters are unchanged from the initial commit: soft limit 15%, hard limit 30%,
  cooldown 20 bars. No other basis, threshold or high-water-mark window will be tried on these
  data.
* v2 becomes the default (`risk.drawdown_basis: shadow`). v1 remains available as
  `drawdown_basis: realized`, so the earlier tables can be reproduced.
* The tests below must pass before any reference config is re-run:
  1. **No trap.** If the strategy recovers its high-water mark, exposure returns to 1.
  2. **Shadow = overlay off.** The shadow equity equals the equity of the same book run
     without any overlay, to floating-point precision.
  3. **Causality.** Scrambling the future leaves past multipliers unchanged.
  4. **Paper parity.** The paper trader keeps its own shadow account and still reproduces the
     backtest to 1e-10.
* **Reporting.** Every reference config is re-run with v2, and v1 / v2 / overlay off are
  published side by side, whatever the outcome. If v2 does worse, that is reported as it is,
  and any later change to the overlay needs a new pre-registration.

**Outcome (added after the run; nothing above was edited).** The four acceptance tests are in
`tests/test_overlay.py` and passed before any reference config was re-run. With
`drawdown_basis: realized`, every earlier result is reproduced exactly (equity gap ≤ 1e-15).

| Portfolio Sharpe (max DD) | v1 | v2 | overlay off |
|---|---:|---:|---:|
| Synthetic multi-strategy | 1.37 (−18.1%) | 1.37 (−18.2%) | 1.38 (−17.8%) |
| Binance, fixed universe | 0.35 (−27.7%) | 0.52 (−32.8%) | 0.53 (−31.2%) |
| Binance, point-in-time | −0.63 (−29.5%) | −0.08 (−42.7%) | −0.15 (−49.9%) |
| Walk-forward ML book | −0.13 (−25.9%) | −0.05 (−29.7%) | 0.19 (−26.9%) |

* The trap is gone: mean exposure in the PIT run goes from 0.24 to 0.89.
* v2 shows no demonstrable value over having no overlay at all. The differences go both ways
  and are within the noise.
* v1's lower drawdowns came from being out of the market. v2 does not cap the book's
  drawdown at 30%, because that limit applies to the shadow book.
* `cost_stress` is unaffected, since it disables the overlay.

## 7. Confidence intervals on published Sharpe ratios

Method fixed before any interval was computed (`alphaforge.analytics.inference`):

* **Circular block bootstrap** (Politis & Romano, 1992). Blocks keep autocorrelation and
  volatility clustering. Block length is `ceil(n^(1/3))`, i.e. 13 bars for the ~1,930 daily
  evaluation bars of the Binance runs; this is a standard rate, not a tuned value.
* **5,000 resamples, 95% two-sided percentile intervals, seed 0.**
* **Differences are paired.** Both books are resampled with the same blocks on their common
  dates, so shared market moves cancel out.
* **Reading rule:** a difference is called meaningful only if its 95% interval excludes zero.
  No multiple-comparison correction is applied to the handful of pre-listed comparisons, so a
  single borderline exclusion should be read with care.
* **Calibration check** (`tests/test_inference.py`): on AR(1) returns with a known Sharpe,
  the interval covers the true value in 88–99% of trials.

Pre-listed comparisons: every portfolio Sharpe in the README; overlay v2 vs v1 and vs no
overlay (all four books); the fixed-universe and point-in-time Binance books vs BTC buy &
hold; fixed vs point-in-time universe.

**Outcome.** Tables are in [RESULTS.md](RESULTS.md#how-much-of-this-is-noise).

* The only Sharpe whose interval excludes zero is the synthetic book (1.37 [0.60, 2.12]).
  No real-data book is distinguishable from zero or from BTC. Intervals are about ±0.8 wide,
  so five years of daily data cannot separate books whose Sharpe differs by less than ~1.
* Fixed vs point-in-time universe: +0.61 [−0.05, 1.30], large but not established.
* Overlay v2 vs v1 is not significant on any book. v2 vs no overlay is significantly negative
  on the ML book (−0.24) and on the synthetic book (−0.01); everywhere else it is not
  significant.

## References

* Bailey, D. & López de Prado, M. (2014). *The Deflated Sharpe Ratio.*
* Benjamini, Y. & Hochberg, Y. (1995). *Controlling the False Discovery Rate.*
* López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
* Moskowitz, T., Ooi, Y. & Pedersen, L. (2012). *Time Series Momentum.*
* Amihud, Y. (2002). *Illiquidity and Stock Returns.*
* Corwin, S. & Schultz, P. (2012). *A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices.*
* Almgren, R. et al. (2005). *Direct Estimation of Equity Market Impact.*
