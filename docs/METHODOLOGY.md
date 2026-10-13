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
| Data snooping | Benjamini-Hochberg + Deflated Sharpe | `alpha.validation` |
| Ignoring carry | Funding charged bar by bar on every position | `BacktestEngine` |

Known residual limitations, stated plainly:

* Outlier repair looks one bar ahead (a bad tick is a spike that reverts). This is acceptable as
  historical hygiene, but a live system could not do it. The threshold is deliberately high (20 MADs).
* The universe is fixed by the config: no point-in-time constituent selection yet. For the
  real-data config this means **survivorship bias**: the 12 perps are today's large caps,
  picked with hindsight, so the result is biased upwards.
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
0.42, against 0.44 for BTC buy & hold, with a −27% drawdown instead of −78% and near-zero
beta. The edges planted in the synthetic market were stronger than reality: funding carry
falls from Sharpe 2.0 to 0.38. One instructive result: `low_vol_60` has the largest rank IC
(t = 8) and a *negative* long-short Sharpe. An IC weights every day equally, while dollar PnL
is dominated by the few high-dispersion days when volatile alts rally. Rank-based evidence
has to be confirmed in PnL terms. Full table and caveats are in the
[README](../README.md#5-real-market-binance-usdt-m-perpetuals).

## References

* Bailey, D. & López de Prado, M. (2014). *The Deflated Sharpe Ratio.*
* Benjamini, Y. & Hochberg, Y. (1995). *Controlling the False Discovery Rate.*
* López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
* Moskowitz, T., Ooi, Y. & Pedersen, L. (2012). *Time Series Momentum.*
* Amihud, Y. (2002). *Illiquidity and Stock Returns.*
* Corwin, S. & Schultz, P. (2012). *A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices.*
* Almgren, R. et al. (2005). *Direct Estimation of Equity Market Impact.*
