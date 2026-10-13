"""Bootstrap confidence intervals for the README's portfolio Sharpe ratios and comparisons.

Usage (Binance rows need the data from the README's "Real data" section)::

    python examples/05_sharpe_confidence.py            # synthetic + Binance
    python examples/05_sharpe_confidence.py --offline  # synthetic configs only

Method and pre-listed comparisons: docs/METHODOLOGY.md section 7.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alphaforge.analytics import sharpe_difference, sharpe_interval
from alphaforge.backtest import BacktestEngine
from alphaforge.config import load_config
from alphaforge.research import run_experiment

ROOT = Path(__file__).resolve().parents[1]
REAL = ["data.end=2026-10-12", "data.source=parquet"]
OVERLAY = {
    "v1": ["risk.drawdown_basis=realized"],
    "v2": [],
    "off": ["risk.max_drawdown=null", "risk.drawdown_soft_limit=null"],
}


def _returns(config: str, overrides: list[str]) -> tuple[pd.Series, object]:
    result = run_experiment(load_config(ROOT / "configs" / config, overrides), save=False)
    return result.portfolio.returns, result


def _btc(result: object, start: pd.Timestamp) -> pd.Series:
    data = result.data  # type: ignore[attr-defined]
    cfg = result.config  # type: ignore[attr-defined]
    btc = next(s for s in data.symbols if s.startswith("BTC"))
    weights = pd.DataFrame(0.0, index=data.index, columns=data.symbols)
    weights[btc] = 1.0
    engine = BacktestEngine.from_config(cfg.backtest, cfg.costs, None)
    return engine.run(weights, data, start=start).returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    books = [
        ("Synthetic multi-strategy", "multi_strategy.yaml", []),
        ("Walk-forward ML book", "ml_walk_forward.yaml", []),
    ]
    if not args.offline:
        books += [
            ("Binance, fixed universe", "binance_perps.yaml", REAL),
            ("Binance, point-in-time", "binance_perps_pit.yaml", REAL),
        ]

    levels, diffs = [], []
    series: dict[tuple[str, str], pd.Series] = {}
    for label, config, base in books:
        results = {}
        for variant, extra in OVERLAY.items():
            r, res = _returns(config, [*base, *extra])
            series[(label, variant)], results[variant] = r, res
            ppy = res.data.periods_per_year  # type: ignore[attr-defined]
            levels.append({"book": label, "overlay": variant, "sharpe": sharpe_interval(r, ppy)})
        for other in ("v1", "off"):
            d = sharpe_difference(series[(label, "v2")], series[(label, other)], ppy)
            diffs.append({"comparison": f"{label}: v2 - {other}", "difference": d})
        if label.startswith("Binance"):
            btc = _btc(results["v2"], series[(label, "v2")].index[0])
            series[(label, "BTC")] = btc
            levels.append(
                {"book": label, "overlay": "BTC buy & hold", "sharpe": sharpe_interval(btc, ppy)}
            )
            d = sharpe_difference(series[(label, "v2")], btc, ppy)
            diffs.append({"comparison": f"{label}: v2 - BTC", "difference": d})
    if not args.offline:
        fixed, pit = (
            series[("Binance, fixed universe", "v2")],
            series[("Binance, point-in-time", "v2")],
        )
        d = sharpe_difference(fixed, pit, 365.0)
        diffs.append({"comparison": "Binance v2: fixed - point-in-time", "difference": d})

    print("Sharpe, 95% block-bootstrap interval")
    for row in levels:
        print(f"  {row['book']:28s} {row['overlay']:15s} {row['sharpe']}")
    print("\nPaired differences (meaningful only if the interval excludes 0)")
    for row in diffs:
        d = row["difference"]
        flag = "excludes 0" if d.excludes_zero else "includes 0"
        print(f"  {row['comparison']:42s} {d}  {flag}")


if __name__ == "__main__":
    main()
