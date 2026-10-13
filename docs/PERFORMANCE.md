# Engine performance

`BacktestEngine.run` is an event loop: one iteration per bar, because path-dependent pieces
(drifted weights, the drawdown overlay, NAV-dependent impact) make each bar depend on the
previous one. This page measures where that loop spends its time and what compiling it buys.

All numbers below come from `scripts/benchmark.py` (seeded synthetic market, seeded random
targets, default engine: square-root impact, one bar of delay, funding, drawdown overlay v2).
They were measured on one Windows 11 laptop with Python 3.14 and numba 0.68. Absolute times
move by tens of percent between runs with the machine's load; the ratios are stable.

## Result

```bash
python scripts/benchmark.py --repeat 5
```

| Assets x bars | Python: time | Python: peak mem | numba: time | numba: peak mem | Speed-up |
|---|---:|---:|---:|---:|---:|
| 12 x 2000 | 263 ms | 1.9 MiB | 8 ms | 2.0 MiB | 31.9x |
| 50 x 2000 | 290 ms | 6.6 MiB | 23 ms | 7.9 MiB | 12.9x |
| 50 x 5000 | 681 ms | 16.4 MiB | 54 ms | 19.6 MiB | 12.5x |
| 200 x 5000 | 890 ms | 63.6 MiB | 208 ms | 77.6 MiB | 4.3x |

- The first numba call in a process costs 0.39 s (loading the compiled loop from the on-disk
  cache). The very first compilation on a machine took 10.1 s, once.
- Peak memory is about 20% higher with numba: the kernel works on contiguous float64 copies of
  the input panels.
- Results are the same to 1e-10 (`tests/test_kernel.py`, a property test over panel size,
  listing gaps, delay, cost model, overlay and leverage; 1,500 random cases were also run
  locally). The Python loop remains the reference implementation and the fallback.

## Why the Python loop was slow

```bash
python scripts/benchmark.py --profile 50x5000
```

| Function | Calls | Share of run (incl. callees) |
|---|---:|---:|
| `_run_python` (alphaforge: engine.py) | 1 | 95% |
| `trade` (alphaforge: engine.py) | 10,000 | 77% |
| `compute` (alphaforge: costs.py, square-root impact) | 10,000 | 71% |
| `nan_to_num` (numpy) | 20,000 | 44% |
| `mark_to_market` (alphaforge: engine.py) | 10,000 | 9% |
| `cost_inputs` (alphaforge: engine.py) | 1 | 4% |

The hot path is the cost model, called twice per bar (real book and shadow book). Most of its
time is not arithmetic on 50 numbers: it is `np.nan_to_num` and the other numpy entry points,
each costing a few microseconds of call overhead on tiny arrays, 20,000 times. Time also grows
with the number of **bars** much more than with the number of **assets** (in the table: 50 to
200 assets at 5,000 bars adds 31%, 2,000 to 5,000 bars at 50 assets adds 135%), the signature
of per-iteration interpreter overhead. Vectorising inside the loop cannot remove that; compiling
the loop can.

The sampling profile shows the same picture:
[engine_flamegraph.svg](img/engine_flamegraph.svg) (`py-spy record --function`; open it in a
browser and click `run` to zoom). The sampler slows the run several times over, so the
flamegraph is read for proportions only, and module imports appear in it because py-spy samples
the whole process.

## Where the time goes now

```bash
python scripts/benchmark.py --profile 200x5000 --kernel numba
```

| Function | Calls | Share of run (incl. callees) |
|---|---:|---:|
| `cost_inputs` (alphaforge: engine.py) | 1 | 58% |
| `_apply` (pandas: rolling.py) | 2 | 34% |
| `_run_kernel` (alphaforge: engine.py, the compiled loop) | 1 | 16% |
| `returns` (alphaforge: market_data.py) | 2 | 9% |

With the loop compiled, the run is dominated by vectorised pandas preparation: EWMA volatility
and rolling dollar volume for the impact model. That is why the speed-up falls as the number of
assets grows.

## Why numba and not Rust

The question was whether a Rust extension (PyO3 + maturin) would justify its cost. After
compilation, the loop is about 16% of a run at 200 x 5000; even an infinitely fast loop would
make the run at most about 1.2x faster. Rust would add a second language, a compiled build in CI
and wheels for every platform, for that margin. numba compiles the same Python source, stays
optional (`pip install "alphaforge[fast]"`) and falls back to the reference loop when absent.

Next lever, if runs need to be faster still: the preparation step (`cost_inputs`), for example
by caching risk inputs across runs that share the same data, as cost and capacity sweeps do.

## When it matters

A single backtest was already under a second. The kernel matters when the engine runs many
times: cost and capacity sweeps, bootstrap or Monte Carlo over paths, parameter studies. At 50
assets over 5,000 bars, 1,000 runs drop from about 11 minutes to about 1 minute.
