"""Command-line interface.

Examples::

    alphaforge run configs/multi_strategy.yaml
    alphaforge run configs/ml_walk_forward.yaml --set portfolio.target_vol=0.15
    alphaforge research configs/multi_strategy.yaml
    alphaforge fetch --exchange binance --symbols BTC/USDT:USDT ETH/USDT:USDT --timeframe 1d
    alphaforge paper configs/multi_strategy.yaml --replay 20
    alphaforge paper configs/binance_perps.yaml
    alphaforge validate configs/binance_perps.yaml
    alphaforge list
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

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
    from alphaforge.data.loader import load_dataset

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
    dataset = load_dataset(config)
    data = dataset.data
    print(f"Stored {len(data.symbols)} symbols x {data.n_bars} bars in {args.store}")
    print(dataset.quality.summary())
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from alphaforge.config.loader import load_config
    from alphaforge.data.loader import load_dataset
    from alphaforge.data.universe import universe_mask

    config = load_config(args.config, args.set)
    data_cfg = config.data.model_copy(
        update={"quality": config.data.quality.model_copy(update={"on_fail": "warn"})}
    )
    dataset = load_dataset(data_cfg, config.cleaning, config.seed)
    table = dataset.quality.table
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.drop(columns=["first_bar", "last_bar"], errors="ignore"))
    members = universe_mask(dataset.data, config.data.universe).sum(axis=1)
    print(
        f"\nUniverse size per bar: min {members.min()}, median {members.median():.0f}, "
        f"max {members.max()}"
    )
    print(dataset.quality.summary())
    if not dataset.quality.passed:
        print("FAILED: error-level flags above would stop a run with data.quality.on_fail: error")
        return 1
    print("OK: no error-level data-quality flags")
    return 0


def _cmd_paper(args: argparse.Namespace) -> int:
    from alphaforge.backtest.engine import BacktestEngine
    from alphaforge.config.loader import load_config
    from alphaforge.data.loader import load_market_data
    from alphaforge.live import PaperTrader
    from alphaforge.utils.logging import configure_logging
    from alphaforge.utils.time import timeframe_to_timedelta

    config = load_config(args.config, args.set)
    configure_logging(config.log_level)

    if args.replay:
        # Offline demo: walk the last N bars through the paper account, each step seeing only
        # data up to that bar, and check it reproduces the backtest over the same bars.
        data, _ = load_market_data(config.data, config.cleaning, config.seed)
        if not 1 <= args.replay < data.n_bars:
            print(f"--replay must be between 1 and {data.n_bars - 1}", file=sys.stderr)
            return 2
        start = data.index[-args.replay]
        trader = PaperTrader(config)
        paper = trader.replay(data, start)
        signals = {k: s.generate_signals(data) for k, s in trader.strategies.items()}
        weights = trader.constructor.build(signals, data).weights
        engine = BacktestEngine.from_config(config.backtest, config.costs, config.risk)
        backtest = engine.run(weights, data, start=start).equity
        diff = float(((paper["nav"] - backtest).abs() / backtest).max())
        table = paper.assign(backtest_nav=backtest, rel_diff=(paper["nav"] - backtest) / backtest)
        with pd.option_context("display.width", 160, "display.max_columns", None):
            print(table.tail(10))
        print(f"\nPaper replay of {len(paper)} bars vs backtest: max relative NAV gap {diff:.2e}")
        if diff > 1e-9:
            print("MISMATCH: live and backtest code paths disagree", file=sys.stderr)
            return 1
        print("OK: the paper loop reproduces the backtest bar for bar.")
        return 0

    if config.data.source == "synthetic":
        print(
            "A paper step needs exchange data (data.source: ccxt). "
            "For an offline demo use --replay N.",
            file=sys.stderr,
        )
        return 2
    data_cfg = config.data
    if data_cfg.source == "ccxt":
        data_cfg = data_cfg.model_copy(update={"use_cache": False})  # always refresh
    data, _ = load_market_data(data_cfg, config.cleaning, config.seed)
    bar = timeframe_to_timedelta(config.data.timeframe)
    age = pd.Timestamp.now(tz="UTC") - (data.index[-1] + bar)
    if age > 2 * bar:
        print(f"WARNING: last complete bar {data.index[-1]} is {age} old", file=sys.stderr)

    state_dir = Path(args.state_dir) / config.name
    trader = PaperTrader.from_state(config, state_dir / "state.json")
    report = trader.step(data)
    if report.skipped:
        print(f"Bar {report.bar} already processed (NAV {report.nav:,.2f}); nothing to do.")
        return 0
    state_dir.mkdir(parents=True, exist_ok=True)
    if len(report.fills):
        fills_path = state_dir / "fills.csv"
        report.fills.to_csv(fills_path, mode="a", header=not fills_path.exists(), index=False)
    equity_path = state_dir / "equity.csv"
    pd.DataFrame(
        [
            {
                "bar": report.bar,
                "nav_before": report.nav_before,
                "nav": report.nav,
                "risk_scale": report.risk_scale,
            }
        ]
    ).to_csv(equity_path, mode="a", header=not equity_path.exists(), index=False)
    trader.save_state(state_dir / "state.json")

    with pd.option_context(
        "display.float_format", "{:,.4f}".format, "display.width", 200, "display.max_columns", None
    ):
        print(report.fills.drop(columns="bar") if len(report.fills) else "No trades")
    print(
        f"\nBar {report.bar}: NAV {report.nav_before:,.2f} -> {report.nav:,.2f} "
        f"(risk scale {report.risk_scale:.2f}). State: {state_dir}"
    )
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

    pp = sub.add_parser(
        "paper", help="Paper-trade a config with the backtest's code (one bar per call)"
    )
    pp.add_argument("config", help="Path to a YAML/JSON experiment config")
    pp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    pp.add_argument(
        "--replay",
        type=int,
        default=0,
        metavar="N",
        help="Offline: replay the last N bars and check parity with the backtest",
    )
    pp.add_argument("--state-dir", default="paper", help="Where paper account state is kept")
    pp.set_defaults(func=_cmd_paper)

    val = sub.add_parser("validate", help="Audit a config's data: quality flags and universe size")
    val.add_argument("config", help="Path to a YAML/JSON experiment config")
    val.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    val.set_defaults(func=_cmd_validate)

    lst = sub.add_parser("list", help="List registered strategies, features and alphas")
    lst.set_defaults(func=_cmd_list)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from alphaforge.config.loader import ConfigError
    from alphaforge.data.quality import DataQualityError
    from alphaforge.data.sources.base import DataSourceError
    from alphaforge.research.experiment import ConfigurationError

    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (
        ConfigError,
        ConfigurationError,
        DataQualityError,
        DataSourceError,
        FileNotFoundError,
    ) as exc:
        # User-facing problems (bad config, missing or bad data, venue down): no traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
