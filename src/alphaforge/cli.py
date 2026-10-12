"""Command-line interface.

Examples::

    alphaforge run configs/multi_strategy.yaml
    alphaforge run configs/ml_walk_forward.yaml --set portfolio.target_vol=0.15
    alphaforge research configs/multi_strategy.yaml
    alphaforge fetch --exchange binance --symbols BTC/USDT:USDT ETH/USDT:USDT --timeframe 1d
    alphaforge list
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import pandas as pd


def _cmd_run(args: argparse.Namespace) -> int:
    from alphaforge.config.loader import load_config
    from alphaforge.research.experiment import run_experiment

    config = load_config(args.config, args.set)
    result = run_experiment(config, save=not args.no_save)
    with pd.option_context(
        "display.float_format", "{:,.3f}".format, "display.width", 160, "display.max_columns", None
    ):
        print(result.summary())
    if result.run_dir is not None:
        print(f"\nReport: {result.run_dir / 'report.html'}")
    return 0


def _cmd_research(args: argparse.Namespace) -> int:
    from alphaforge.config.loader import load_config
    from alphaforge.research.experiment import run_alpha_research

    config = load_config(args.config, args.set)
    report, run_dir = run_alpha_research(config, save=not args.no_save)
    with pd.option_context(
        "display.float_format", "{:,.3f}".format, "display.width", 200, "display.max_columns", None
    ):
        print(report.leaderboard)
    if run_dir is not None:
        print(f"\nResults: {run_dir}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from alphaforge.config.schema import DataConfig
    from alphaforge.data.loader import load_market_data

    config = DataConfig(
        source="ccxt",
        exchange=args.exchange,
        market_type=args.market_type,
        symbols=args.symbols,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        storage_dir=args.store,
        use_cache=False,
    )
    data, _ = load_market_data(config)
    print(f"Stored {len(data.symbols)} symbols x {data.n_bars} bars in {args.store}")
    return 0


def _cmd_list(_: argparse.Namespace) -> int:
    from alphaforge.alpha.library import default_alpha_library
    from alphaforge.features import list_features
    from alphaforge.strategies import available_strategies

    print("Strategies:")
    for name, cls in sorted(available_strategies().items()):
        fit = " (requires walk-forward)" if cls.requires_fit else ""
        print(f"  {name:20s} {cls.family}{fit}")
    print("\nFeatures:")
    for f in list_features():
        print(f"  {f.name:20s} [{f.category}] {f.description}")
    print("\nAlphas:")
    for a in default_alpha_library():
        print(f"  {a.name:22s} [{a.category}] {a.description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alphaforge", description="AlphaForge quant research CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn, help_ in (
        ("run", _cmd_run, "Run a backtest experiment from a config file"),
        ("research", _cmd_research, "Evaluate the alpha library on a config's data"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("config", help="Path to a YAML/JSON experiment config")
        p.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Override a config value, e.g. --set portfolio.target_vol=0.15",
        )
        p.add_argument("--no-save", action="store_true", help="Do not write a run directory")
        p.set_defaults(func=fn)

    f = sub.add_parser("fetch", help="Download exchange data into the local Parquet store")
    f.add_argument("--exchange", default="binance")
    f.add_argument("--market-type", default="swap", choices=["spot", "swap"])
    f.add_argument("--symbols", nargs="+", required=True)
    f.add_argument("--timeframe", default="1d")
    f.add_argument("--start", default="2021-01-01")
    f.add_argument("--end", default=None)
    f.add_argument("--store", default="data/store")
    f.set_defaults(func=_cmd_fetch)

    lst = sub.add_parser("list", help="List registered strategies, features and alphas")
    lst.set_defaults(func=_cmd_list)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
