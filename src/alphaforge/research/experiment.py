"""Config-driven experiment orchestration with full provenance.

``run_experiment(config)`` = load data -> build strategies -> (walk-forward) signals ->
portfolio construction -> backtest with costs/funding/risk overlay -> per-sleeve backtests ->
report + manifest. Every run gets its own directory with the resolved config, a manifest
(config hash, data hash, git commit, package versions, seed) and the log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from alphaforge.alpha.research import AlphaResearcher, AlphaResearchReport
from alphaforge.backtest.engine import BacktestEngine
from alphaforge.backtest.results import BacktestResult
from alphaforge.backtest.walk_forward import WalkForwardRunner
from alphaforge.config.loader import config_to_dict
from alphaforge.config.schema import ExperimentConfig
from alphaforge.data.cleaning import CleaningReport
from alphaforge.data.loader import load_market_data
from alphaforge.data.market_data import MarketData
from alphaforge.portfolio.construction import PortfolioConstructor, PortfolioTargets
from alphaforge.strategies import Strategy, build_strategy
from alphaforge.utils.logging import configure_logging, get_logger
from alphaforge.utils.reproducibility import (
    RunManifest,
    create_run_dir,
    git_commit,
    hash_dataframe,
    hash_object,
    set_global_seed,
)

logger = get_logger(__name__)


class ConfigurationError(ValueError):
    """Raised when a configuration would produce a methodologically invalid experiment."""


@dataclass
class ExperimentResult:
    """Everything an experiment produced."""

    config: ExperimentConfig
    data: MarketData
    portfolio: BacktestResult
    sleeves: dict[str, BacktestResult]
    targets: PortfolioTargets
    cleaning: list[CleaningReport]
    run_dir: Path | None = None
    fold_metrics: pd.DataFrame | None = None
    extras: dict[str, pd.DataFrame] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Key metrics of the portfolio and each sleeve."""
        cols = [
            "cagr",
            "volatility",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "annual_turnover",
        ]
        rows = {self.portfolio.name: self.portfolio.metrics().to_series()}
        rows.update({k: v.metrics().to_series() for k, v in self.sleeves.items()})
        return pd.DataFrame(rows).T[cols]


def _strategy_factory(config: ExperimentConfig) -> dict[str, Strategy]:
    return {spec.label: build_strategy(spec.name, spec.params) for spec in config.strategies}


def run_experiment(
    config: ExperimentConfig,
    data: MarketData | None = None,
    save: bool | None = None,
) -> ExperimentResult:
    """Run a full experiment.

    Args:
        config: Validated experiment configuration.
        data: Pre-loaded market data (skips the data layer; useful in notebooks/tests).
        save: Override ``config.report.save``.

    Raises:
        ConfigurationError: if a strategy that must be fitted is run without walk-forward.
    """
    save = config.report.save if save is None else save
    run_dir = create_run_dir(config.report.output_dir, config.name) if save else None
    configure_logging(config.log_level, run_dir / "run.log" if run_dir else None)
    set_global_seed(config.seed)
    logger.info("Experiment %r (seed=%d)", config.name, config.seed)

    cleaning: list[CleaningReport] = []
    if data is None:
        data, cleaning = load_market_data(config.data, config.cleaning, config.seed)

    strategies = _strategy_factory(config)
    needs_fit = [label for label, s in strategies.items() if s.requires_fit]
    if needs_fit and not config.walk_forward.enabled:
        raise ConfigurationError(
            f"Strategies {needs_fit} learn from data and must be evaluated out-of-sample: "
            "set walk_forward.enabled: true"
        )

    constructor = PortfolioConstructor(config.portfolio)
    engine = BacktestEngine.from_config(config.backtest, config.costs, config.risk)
    sleeve_engine = BacktestEngine.from_config(config.backtest, config.costs, risk=None)

    fold_metrics = None
    if config.walk_forward.enabled:
        wf = WalkForwardRunner(
            lambda: _strategy_factory(config),
            constructor,
            engine,
            config.walk_forward.train_bars,
            config.walk_forward.test_bars,
            config.walk_forward.expanding,
            config.walk_forward.embargo_bars,
        ).run(data)
        portfolio, targets, start = wf.backtest, wf.targets, wf.oos_start
        portfolio.name = "portfolio"
        fold_metrics = wf.fold_metrics
    else:
        signals = {label: s.generate_signals(data) for label, s in strategies.items()}
        targets = constructor.build(signals, data)
        start = config.backtest.start
        portfolio = engine.run(targets.weights, data, start=start, name="portfolio")

    sleeves = {
        label: sleeve_engine.run(w, data, start=start, name=label)
        for label, w in targets.sleeve_weights.items()
    }
    result = ExperimentResult(
        config, data, portfolio, sleeves, targets, cleaning, run_dir, fold_metrics
    )
    logger.info(
        "Portfolio: %s", {k: round(v, 3) for k, v in result.summary().loc["portfolio"].items()}
    )

    if run_dir is not None:
        _persist(result)
    return result


def _persist(result: ExperimentResult) -> None:
    from alphaforge.analytics.report import generate_report

    config, run_dir = result.config, result.run_dir
    assert run_dir is not None
    cfg_dict = config_to_dict(config)
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
    RunManifest(
        run_id=run_dir.name,
        experiment=config.name,
        seed=config.seed,
        config_hash=hash_object(cfg_dict),
        data_hash=hash_dataframe(result.data.close),
        git_commit=git_commit(),
    ).to_json(run_dir / "manifest.json")
    if result.cleaning:
        pd.DataFrame([r.to_dict() for r in result.cleaning]).to_csv(
            run_dir / "cleaning_report.csv", index=False
        )

    extra = {}
    if result.fold_metrics is not None:
        extra["walk_forward_folds"] = result.fold_metrics
    result.targets.allocation.to_csv(run_dir / "allocation.csv")
    generate_report(
        result.portfolio,
        run_dir,
        title=f"AlphaForge · {config.name}",
        sleeves=result.sleeves,
        allocation=result.targets.allocation,
        extra_tables=extra,
        config=cfg_dict,
        plots=config.report.plots,
        html=config.report.html,
    )
    logger.info("Report written to %s", run_dir)


def run_alpha_research(
    config: ExperimentConfig, data: MarketData | None = None, save: bool | None = None
) -> tuple[AlphaResearchReport, Path | None]:
    """Evaluate the default alpha library on the configured data and save a leaderboard."""
    save = config.report.save if save is None else save
    run_dir = create_run_dir(config.report.output_dir, f"{config.name}_alphas") if save else None
    configure_logging(config.log_level, run_dir / "run.log" if run_dir else None)
    set_global_seed(config.seed)
    if data is None:
        data, _ = load_market_data(config.data, config.cleaning, config.seed)
    rc = config.research
    report = AlphaResearcher(horizons=rc.horizons, n_quantiles=rc.n_quantiles, fdr=rc.fdr).evaluate(
        data
    )
    if run_dir is not None:
        from alphaforge.analytics.plots import plot_correlation_heatmap, plot_ic_decay

        report.leaderboard.to_csv(run_dir / "alpha_leaderboard.csv")
        report.ic_decay().to_csv(run_dir / "ic_decay.csv")
        plot_ic_decay(report.ic_decay()).savefig(
            run_dir / "ic_decay.png", dpi=110, bbox_inches="tight"
        )
        top = report.leaderboard.index[: max(rc.top_k, 2)]
        plot_correlation_heatmap(
            report.long_short_returns[top], "Alpha long-short correlation"
        ).savefig(run_dir / "alpha_correlation.png", dpi=110, bbox_inches="tight")
        logger.info("Alpha research written to %s", run_dir)
    return report, run_dir
