"""Config validation and end-to-end experiment runs."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from alphaforge.cli import main
from alphaforge.config import ExperimentConfig, apply_overrides, load_config
from alphaforge.research import ConfigurationError, run_experiment

CONFIGS = sorted(Path(__file__).resolve().parents[1].joinpath("configs").glob("*.yaml"))

SMALL = {
    "name": "test",
    "data": {"symbols": ["A", "B", "C", "D", "E", "F"], "synthetic": {"n_bars": 400}},
    "strategies": [{"name": "ts_momentum"}, {"name": "funding_carry"}],
    "backtest": {"start": "2020-03-01"},
    "report": {"save": False},
}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_validate(path: Path) -> None:
    assert isinstance(load_config(path), ExperimentConfig)


def test_config_rejects_typos_and_duplicates() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({**SMALL, "portfolio": {"target_volatility": 0.2}})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({**SMALL, "strategies": [{"name": "ts_momentum"}] * 2})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({**SMALL, "data": {**SMALL["data"], "timeframe": "7m"}})


def test_overrides() -> None:
    raw = apply_overrides(SMALL, ["portfolio.target_vol=0.1", "seed=7"])
    cfg = ExperimentConfig.model_validate(raw)
    assert cfg.portfolio.target_vol == 0.1 and cfg.seed == 7


def test_fitted_strategy_requires_walk_forward() -> None:
    cfg = ExperimentConfig.model_validate({**SMALL, "strategies": [{"name": "ml_directional"}]})
    with pytest.raises(ConfigurationError):
        run_experiment(cfg)


def test_run_experiment_in_memory() -> None:
    result = run_experiment(ExperimentConfig.model_validate(SMALL))
    assert set(result.sleeves) == {"ts_momentum", "funding_carry"}
    assert result.portfolio.returns.index[0].strftime("%Y-%m-%d") == "2020-03-01"
    summary = result.summary()
    assert list(summary.index) == ["portfolio", "ts_momentum", "funding_carry"]


@pytest.mark.slow
def test_walk_forward_experiment_writes_report(tmp_path: Path) -> None:
    raw = {
        **SMALL,
        "data": {**SMALL["data"], "synthetic": {"n_bars": 600}},
        "strategies": [
            {"name": "ml_directional", "params": {"model": "logistic"}},
            {"name": "ts_momentum"},
        ],
        "walk_forward": {"enabled": True, "train_bars": 300, "test_bars": 100, "embargo_bars": 5},
        "backtest": {},
        "report": {"save": True, "output_dir": str(tmp_path)},
    }
    result = run_experiment(ExperimentConfig.model_validate(raw))
    assert result.fold_metrics is not None and len(result.fold_metrics) == 3
    run_dir = result.run_dir
    assert run_dir is not None
    for name in ("report.html", "manifest.json", "config.json", "metrics.csv", "summary.md"):
        assert (run_dir / name).exists(), name
    # out-of-sample only: first traded bar is after the first training window + embargo
    assert result.portfolio.returns.index[0] == result.data.index[305]


def test_cli_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "ts_momentum" in out and "funding_rate" in out and "mom_60" in out
