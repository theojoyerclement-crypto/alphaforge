# Architecture

## Design principles

1. **Causality is a tested property, not a convention.** Every feature and rule-based strategy
   is run twice, once on the real data and once with the future scrambled, and the past must be
   bit-identical. A new feature or strategy is covered automatically through the registries.
2. **Separation of concerns along the research pipeline.** A strategy only says *which
   direction and how convinced* (`[-1, 1]` per asset). Sizing, risk and allocation belong to
   `portfolio`; frictions and accounting to `backtest`. This lets a strategy be evaluated
   stand-alone, inside a book, or under different cost assumptions without being touched.
3. **Wide panels everywhere.** `MarketData` holds aligned `time × symbol` DataFrames. Cross-
   sectional operations are row-wise, time-series operations column-wise, and both vectorise.
   The ML layer converts to a long `(timestamp, symbol)` design matrix only at the boundary.
4. **Configuration as the experiment's identity.** A run is fully determined by its YAML file
   (validated by pydantic with `extra="forbid"`) plus the data; both are hashed into the manifest.
5. **Honest defaults.** One bar of execution delay, square-root impact, funding on, drawdown
   control on. Turning realism *off* requires an explicit override.

## Data flow

```
DataConfig ──> source (synthetic | ccxt | parquet)
            ──> clean_ohlcv (per symbol, CleaningReport)
            ──> aggregate_funding_to_bars
            ──> MarketData(open, high, low, close, volume, funding, extras={basis})

MarketData ──> Strategy.generate_signals ──> {label: signals}          (fit() on train only)
           ──> PortfolioConstructor.build
                 size_sleeve:  signal / asset_vol  ──> sleeve vol target (EWMA covariance)
                 StrategyAllocator: ERC on trailing sleeve returns, correlation haircut
                 portfolio vol target ──> position / gross / net limits
           ──> BacktestEngine.run (event loop)
                 mark-to-market ──> funding ──> drift ──> drawdown overlay ──> trade ──> costs
           ──> BacktestResult ──> metrics, attribution, report
```

## Timing conventions

| Quantity | Known at | Used for |
|---|---|---|
| Bar *t* OHLCV, funding accrued during *t* | close of *t* | features at *t* |
| Signal / target at *t* | close of *t* | execution at close of *t + delay* |
| Position held over bar *t+1* | close of *t* | earns `r(t+1)` and pays `funding(t+1)` |
| Risk estimates (EWMA vol, covariance) at *t* | close of *t* | sizing decided at *t* |
| Strategy allocation at *t* | close of *t* | uses sleeve returns up to *t* only |
| Forward returns | future | **alpha evaluation only**, never an input |

## Engine accounting

Per bar, with `w` the post-trade weights of the previous close:

```
price_pnl   = w · r_t
funding_pnl = −w · f_t                  (longs pay positive funding)
w_drift     = w ⊙ (1 + r_t) / (1 + price_pnl + funding_pnl)
scale       = DrawdownController.update(NAV_pre)
trades      = target_{t−delay} · scale − w_drift
cost        = Σ |trades| (fee + half_spread) + |trades| · k · σ · sqrt(|trades| · NAV / ADV)
net_return  = price_pnl + funding_pnl − cost · (1 + price_pnl + funding_pnl)
```

The identity `net = gross + funding − costs` holds exactly and is asserted in the tests.

## Extension points

| To add | Implement | Register with |
|---|---|---|
| Feature | `f(data, **params) -> DataFrame` | `@register_feature(name, category)` |
| Alpha | `Alpha(name, expression, category)` | add to `default_alpha_library()` or pass a list |
| Strategy | subclass `Strategy`, implement `generate_signals` (and `fit` if it learns) | `@register_strategy` |
| Cost model | subclass `CostModel.compute` | `build_cost_model` |
| Data source | subclass `DataSource.fetch_ohlcv/fetch_funding` | `data.loader` |

## Module dependency order

`utils → config → data → features → alpha → portfolio → analytics.metrics → backtest → ml →
strategies → research → cli`. `backtest.walk_forward` references strategies only for typing,
which keeps the graph acyclic; `tests/test_imports.py` imports each package in a fresh
interpreter to guard against cycles.
