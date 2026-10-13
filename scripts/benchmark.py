"""Reproducible benchmark of BacktestEngine.run: wall time and peak memory by panel size.

    python scripts/benchmark.py                       # markdown table on stdout
    python scripts/benchmark.py --sizes 50x5000 --repeat 3
    python scripts/benchmark.py --profile 50x5000     # cProfile: where engine time goes
    py-spy record -o docs/img/engine_flamegraph.svg -- python scripts/benchmark.py --sizes 50x5000

Inputs are deterministic (seeded synthetic market, seeded random targets) and the engine runs
with the default configuration: square-root impact, one bar of delay, funding, drawdown overlay
v2 (shadow book). Time is the median of ``--repeat`` runs; peak memory is the tracemalloc peak
of one run (numpy allocations included). Only the engine is timed, not data loading.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alphaforge.backtest import BacktestEngine  # noqa: E402
from alphaforge.config.schema import (  # noqa: E402
    BacktestConfig,
    CostConfig,
    DataConfig,
    RiskConfig,
    SyntheticConfig,
)
from alphaforge.data.loader import load_market_data  # noqa: E402
from alphaforge.data.market_data import MarketData  # noqa: E402

DEFAULT_SIZES = ("12x2000", "50x2000", "50x5000", "200x5000")


def make_inputs(n_assets: int, n_bars: int, seed: int = 0) -> tuple[pd.DataFrame, MarketData]:
    """Seeded synthetic market and random target weights (zero where untradable)."""
    symbols = [f"S{i:03d}" for i in range(n_assets)]
    config = DataConfig(symbols=symbols, synthetic=SyntheticConfig(n_bars=n_bars))
    data, _ = load_market_data(config, seed=seed)
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0 / n_assets, size=(n_bars, n_assets))
    weights = pd.DataFrame(raw, index=data.index, columns=data.symbols)
    return weights.where(data.close.notna(), 0.0), data


def default_engine(**kwargs: object) -> BacktestEngine:
    engine = BacktestEngine.from_config(BacktestConfig(), CostConfig(), RiskConfig())
    for key, value in kwargs.items():
        setattr(engine, key, value)
    return engine


def measure(fn: Callable[[], object], repeat: int) -> tuple[float, float]:
    """``(median seconds, peak MiB)``."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return statistics.median(times), peak / 2**20


def profile(size: str) -> None:
    """Print each hot function's share of ``BacktestEngine.run`` (cProfile, own + callees)."""
    import cProfile
    import pstats

    n_assets, n_bars = (int(x) for x in size.lower().split("x"))
    weights, data = make_inputs(n_assets, n_bars)
    engine = default_engine()
    profiler = cProfile.Profile()
    profiler.runcall(engine.run, weights, data)
    stats = pstats.Stats(profiler).stats  # type: ignore[attr-defined]
    total = next(v[3] for k, v in stats.items() if k[2] == "run" and k[0].endswith("engine.py"))
    print(f"BacktestEngine.run on {n_assets} x {n_bars}: {total * 1000:,.0f} ms under cProfile")
    print("| Function | Calls | Share of run (incl. callees) |")
    print("|---|---:|---:|")
    rows = sorted(stats.items(), key=lambda kv: -kv[1][3])
    for (file, _, name), (_, calls, _, cumulative, _) in rows:
        if 0.03 < cumulative / total < 0.999 and name != "<lambda>":
            where = Path(file).name if "alphaforge" in file else f"numpy {Path(file).name}"
            print(f"| `{name}` ({where}) | {calls:,} | {cumulative / total:.0%} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", default=list(DEFAULT_SIZES), metavar="ASSETSxBARS")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--profile", metavar="ASSETSxBARS", help="cProfile one run instead")
    args = parser.parse_args()
    if args.profile:
        profile(args.profile)
        return

    print("| Assets x bars | Python reference: time | peak memory |")
    print("|---|---:|---:|")
    for size in args.sizes:
        n_assets, n_bars = (int(x) for x in size.lower().split("x"))
        weights, data = make_inputs(n_assets, n_bars)
        engine = default_engine()
        seconds, mib = measure(lambda e=engine, w=weights, d=data: e.run(w, d), args.repeat)
        print(f"| {n_assets} x {n_bars} | {seconds * 1000:,.0f} ms | {mib:,.1f} MiB |", flush=True)


if __name__ == "__main__":
    main()
