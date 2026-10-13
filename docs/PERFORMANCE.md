# Engine performance

`BacktestEngine.run` is an event loop: one Python iteration per bar, because path-dependent
pieces (drifted weights, the drawdown overlay, NAV-dependent impact) make each bar depend on the
previous one. This page measures where that loop spends its time.

All numbers below come from `scripts/benchmark.py` (seeded synthetic market, seeded random
targets, default engine: square-root impact, one bar of delay, funding, drawdown overlay v2).
They were measured on one Windows 11 laptop with Python 3.14; absolute times vary by machine,
ratios much less.

## Baseline: the Python reference

```bash
python scripts/benchmark.py --repeat 3
```

| Assets x bars | Python reference: time | peak memory |
|---|---:|---:|
| 12 x 2000 | 208 ms | 1.9 MiB |
| 50 x 2000 | 227 ms | 6.6 MiB |
| 50 x 5000 | 569 ms | 16.4 MiB |
| 200 x 5000 | 904 ms | 63.6 MiB |

Time grows with the number of **bars** much more than with the number of **assets**: going from
50 to 200 assets at 5,000 bars adds 59%, while going from 2,000 to 5,000 bars at 50 assets adds
151%. That is the signature of per-iteration interpreter overhead rather than arithmetic.

## Where the time goes

```bash
python scripts/benchmark.py --profile 50x5000
```

| Function | Calls | Share of run (incl. callees) |
|---|---:|---:|
| `trade` (engine.py) | 10,000 | 77% |
| `compute` (costs.py, square-root impact) | 10,000 | 71% |
| `nan_to_num` (numpy) | 20,000 | 44% |
| `mark_to_market` (engine.py) | 10,000 | 8% |
| `cost_inputs` (engine.py, vectorised, once) | 1 | 5% |

The hot path is the cost model, called twice per bar (real book and shadow book). Most of its
time is not arithmetic on 50 numbers: it is `np.nan_to_num` and the other numpy entry points,
each costing a few microseconds of call overhead on tiny arrays, 20,000 times.

The sampling profile (`py-spy record --function`) shows the same picture:
[engine_flamegraph.svg](img/engine_flamegraph.svg) (open it in a browser and click `run` to
zoom). Under the sampler the run is several times slower, so the flamegraph is only read for
proportions, and module imports appear in it because py-spy samples the whole process.

## What this means in practice

A single backtest is not slow: under a second for 200 assets over 5,000 daily bars. A faster
kernel matters when the engine runs many times: cost and capacity sweeps, bootstrap or Monte
Carlo over paths, and parameter studies, where hundreds of runs are common.
