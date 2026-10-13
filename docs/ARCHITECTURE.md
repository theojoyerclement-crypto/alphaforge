# Architecture

## Design choices that matter to a hiring manager

| Choice | Consequence | Proof |
|---|---|---|
| **Causality is enforced by tests, not by reviews.** Every registered feature and strategy, and the portfolio constructor, is re-run with the future scrambled. | A new strategy cannot silently peek ahead: the existing tests fail. | `tests/test_features.py`, `tests/test_strategies.py`, `tests/test_live.py` |
| **Strategies emit direction only** (`[-1, 1]`). Sizing, vol targeting, allocation, costs and funding live downstream. | The same signal can be scored alone, inside a book, or under harsher frictions without editing it. | `strategies/base.py`, `portfolio/construction.py` |
| **Frictions are on by default**: 1 bar of latency, fees, spread, square-root impact and perp funding. Turnover is measured on drifted weights. | Turning realism *off* takes an explicit override, and the cost-stress config shows what that hides. | `backtest/engine.py`, `configs/cost_stress.yaml` |
| **One accounting function for backtest and live.** `mark_to_market` and the cost model are shared by `BacktestEngine` and `PaperBroker`. | Replaying history through the paper account reproduces backtest equity to 1e-10. | `live/`, `tests/test_live.py` |
| **Learning strategies can only run under walk-forward.** | No in-sample ML result can be produced by accident. | `research/experiment.py` |
| **A run is identified by its config.** The config is a strict pydantic schema, and the manifest records config, data and code hashes, a dirty-tree flag, the seed and package versions. | Any number in the README can be traced back to an exact input. | `config/schema.py`, `utils/reproducibility.py` |
| **Layered packages with an import-cycle guard.** | Each layer can be tested and replaced on its own. | `tests/test_imports.py`, [dependency order](#module-dependency-order) |

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

Live (one call per closed bar):
MarketData[≤ t] ──> same strategies + PortfolioConstructor ──> target decided at t − delay
                ──> Broker.sync (mark to market, shared mark_to_market) ──> drawdown overlay
                ──> Broker.execute (PaperBroker: engine cost model) ──> state.json / fills.csv
```

The live loop needs no separate signal code. It takes row `−1 − delay` of a target panel
computed from data up to now, and that row equals what was decided `delay` bars ago only
because everything upstream is causal. `PaperTrader.replay` checks this end to end against
`BacktestEngine`.

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
| Venue / broker | subclass `live.Broker` (`nav`, `weights`, `last_bar`, `sync`, `execute`) | pass to `PaperTrader(config, broker=...)` |

## Module dependency order

`utils → config → data → features → alpha → portfolio → analytics.metrics → backtest → ml →
strategies → research → live → cli`. `backtest.walk_forward` references strategies only for typing,
which keeps the graph acyclic; `tests/test_imports.py` imports each package in a fresh
interpreter to guard against cycles.
