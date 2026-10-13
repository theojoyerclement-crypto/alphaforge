"""Reproducible benchmark of BacktestEngine.run: wall time and peak memory by panel size.

    python scripts/benchmark.py                       # markdown table on stdout
    python scripts/benchmark.py --sizes 50x5000 --repeat 3
    python scripts/benchmark.py --profile 50x5000     # cProfile: where engine time goes
    python scripts/benchmark.py --profile 200x5000 --kernel numba
    py-spy record -o docs/img/engine_flamegraph.svg -- python scripts/benchmark.py --sizes 50x5000

Inputs are deterministic (seeded synthetic market, seeded random targets) and the engine runs
with the default configuration: square-root impact, one bar of delay, funding, drawdown overlay
v2 (shadow book). The Python reference loop and the numba kernel (if installed) are timed on
the same inputs. Time is the median of ``--repeat`` runs after warm-up; peak memory is the
tracemalloc peak of one run (numpy allocations included). Only the engine is timed, not data
loading. The numba first-call cost (compilation, or loading the on-disk cache) is reported
separately.
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


def time_engine(
    kernel: str, weights: pd.DataFrame, data: MarketData, repeat: int
) -> tuple[float, float]:
    """``(median seconds, peak MiB)`` of ``BacktestEngine.run`` with the given loop."""
    engine = default_engine(kernel=kernel)
    return measure(lambda: engine.run(weights, data), repeat)


def profile(size: str, kernel: str = "python") -> None:
    """Print each hot function's share of ``BacktestEngine.run`` (cProfile, own + callees)."""
    import cProfile
    import pstats

    n_assets, n_bars = (int(x) for x in size.lower().split("x"))
    weights, data = make_inputs(n_assets, n_bars)
    engine = default_engine(kernel=kernel)
    engine.run(weights, data)  # warm-up: exclude numba compilation from the profile
    profiler = cProfile.Profile()
    profiler.runcall(engine.run, weights, data)
    stats = pstats.Stats(profiler).stats  # type: ignore[attr-defined]
    total = next(v[3] for k, v in stats.items() if k[2] == "run" and k[0].endswith("engine.py"))
    print(f"BacktestEngine.run ({kernel}) on {n_assets} x {n_bars}: {total * 1000:,.0f} ms")
    print("| Function | Calls | Share of run (incl. callees) |")
    print("|---|---:|---:|")
    rows = sorted(stats.items(), key=lambda kv: -kv[1][3])
    shown = 0
    for (file, _, name), (_, calls, _, cumulative, _) in rows:
        share = cumulative / total
        parts = Path(file).parts
        library = next((p for p in ("alphaforge", "pandas", "numpy") if p in parts), None)
        # alphaforge's own functions, plus the single biggest third-party entry point
        if library is None or not 0.03 < share < 0.999:
            continue
        if library != "alphaforge":
            if shown:
                continue
            shown += 1
        print(f"| `{name}` ({library}: {Path(file).name}) | {calls:,} | {share:.0%} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", default=list(DEFAULT_SIZES), metavar="ASSETSxBARS")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--profile", metavar="ASSETSxBARS", help="cProfile one run instead")
    parser.add_argument("--kernel", default="python", choices=["python", "numba"])
    args = parser.parse_args()
    if args.profile:
        profile(args.profile, args.kernel)
        return

    from alphaforge.backtest import _kernel

    if _kernel.AVAILABLE:  # first call in this process: compile, or load the on-disk cache
        warm_w, warm_d = make_inputs(4, 100)
        start = time.perf_counter()
        default_engine(kernel="numba").run(warm_w, warm_d)
        print(f"numba first call (compile or cache load): {time.perf_counter() - start:.2f} s")
        print()

    print(
        "| Assets x bars | Python: time | Python: peak mem | numba: time | numba: peak mem "
        "| Speed-up |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for size in args.sizes:
        n_assets, n_bars = (int(x) for x in size.lower().split("x"))
        weights, data = make_inputs(n_assets, n_bars)
        py_s, py_mib = time_engine("python", weights, data, args.repeat)
        row = f"| {n_assets} x {n_bars} | {py_s * 1000:,.0f} ms | {py_mib:,.1f} MiB "
        if _kernel.AVAILABLE:
            nb_s, nb_mib = time_engine("numba", weights, data, args.repeat)
            row += f"| {nb_s * 1000:,.0f} ms | {nb_mib:,.1f} MiB | {py_s / nb_s:,.1f}x |"
        else:
            row += "| n/a | n/a | n/a |"
        print(row, flush=True)


if __name__ == "__main__":
    main()
