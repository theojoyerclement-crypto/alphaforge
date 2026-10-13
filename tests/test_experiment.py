"""Config validation and end-to-end experiment runs."""

from __future__ import annotations

import json
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
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert {"git_commit", "git_dirty", "config_hash", "data_hash", "seed"} <= set(manifest)
    assert manifest["environment"]["packages"]["alphaforge"] is not None
    # out-of-sample only: first traded bar is after the first training window + embargo
    assert result.portfolio.returns.index[0] == result.data.index[305]


def test_cli_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "ts_momentum" in out and "funding_rate" in out and "mom_60" in out


def test_config_errors_are_readable(tmp_path: Path) -> None:
    import yaml

    from alphaforge.config import ConfigError

    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("name: x\ndata: {symbols: [A, B]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(broken)
    typo = tmp_path / "typo.yaml"
    typo.write_text(yaml.safe_dump({**SMALL, "portfolio": {"target_volatility": 0.2}}), "utf-8")
    with pytest.raises(
        ConfigError, match=r"portfolio\.target_volatility: unknown key .*target_vol"
    ):
        load_config(typo)


def test_strategy_mistakes_fail_before_loading_data() -> None:
    bad_name = ExperimentConfig.model_validate({**SMALL, "strategies": [{"name": "ts_momentun"}]})
    with pytest.raises(ConfigurationError, match=r"strategies\[0\].*available"):
        run_experiment(bad_name)
    bad_param = ExperimentConfig.model_validate(
        {**SMALL, "strategies": [{"name": "ts_momentum", "params": {"lookback": 5}}]}
    )
    with pytest.raises(ConfigurationError, match=r"accepted: .*lookbacks"):
        run_experiment(bad_param)


def test_cli_prints_config_errors_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", str(tmp_path / "missing.yaml")]) == 2
    assert capsys.readouterr().err.startswith("error: Config file not found")


def test_unquoted_yaml_dates_are_accepted() -> None:
    raw = apply_overrides(SMALL, ["data.end=2020-06-30", "backtest.start=2020-03-01"])
    config = ExperimentConfig.model_validate(raw)
    assert config.data.end == "2020-06-30" and config.backtest.start == "2020-03-01"
