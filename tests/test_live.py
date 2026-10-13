"""Paper trading must run the backtest's code: same targets, same frictions, same equity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaforge.backtest.engine import BacktestEngine
from alphaforge.cli import main
from alphaforge.config import ExperimentConfig
from alphaforge.data.loader import load_market_data
from alphaforge.data.market_data import MarketData
from alphaforge.live import AccountState, PaperBroker, PaperTrader
from alphaforge.research import ConfigurationError

BASE = {
    "name": "paper_test",
    "data": {"symbols": ["A", "B", "C", "D", "E"], "synthetic": {"n_bars": 260}},
    "strategies": [{"name": "ts_momentum"}, {"name": "funding_carry"}],
    "portfolio": {"allocation_min_history": 20, "allocation_lookback": 40},
    # Tight limits so the drawdown overlay actually bites during the replay window.
    "risk": {"max_drawdown": 0.06, "drawdown_soft_limit": 0.02, "cooldown_bars": 3},
    "report": {"save": False},
}
REPLAY = 15


def _config(**overrides: object) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**BASE, **overrides})


def _data(config: ExperimentConfig) -> MarketData:
    data, _ = load_market_data(config.data, config.cleaning, config.seed)
    return data


def _backtest_equity(config: ExperimentConfig, data: MarketData, start: pd.Timestamp) -> pd.Series:
    trader = PaperTrader(config)
    signals = {k: s.generate_signals(data) for k, s in trader.strategies.items()}
    weights = trader.constructor.build(signals, data).weights
    engine = BacktestEngine.from_config(config.backtest, config.costs, config.risk)
    return engine.run(weights, data, start=start).equity


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_replay_reproduces_backtest_equity(delay: int) -> None:
    config = _config(backtest={"execution_delay": delay})
    data = _data(config)
    start = data.index[-REPLAY]
    paper = PaperTrader(config).replay(data, start)
    expected = _backtest_equity(config, data, start)
    assert len(paper) == REPLAY
    np.testing.assert_allclose(paper["nav"].to_numpy(), expected.to_numpy(), rtol=1e-10)
    assert paper["turnover"].sum() > 0


def test_replay_exercises_the_drawdown_overlay() -> None:
    config = _config()
    data = _data(config)
    paper = PaperTrader(config).replay(data, data.index[-60])
    assert (paper["risk_scale"] < 1).any(), "pick limits that trigger the overlay"
    np.testing.assert_allclose(
        paper["nav"].to_numpy(),
        _backtest_equity(config, data, data.index[-60]).to_numpy(),
        rtol=1e-10,
    )


def test_resume_from_saved_state_matches_uninterrupted_run(tmp_path: Path) -> None:
    config = _config()
    data = _data(config)
    start = data.index[-REPLAY]
    uninterrupted = PaperTrader(config).replay(data, start)

    state = tmp_path / "state.json"
    split = data.n_bars - REPLAY // 2
    first = PaperTrader(config)
    first.replay(data.iloc(0, split), start)
    first.save_state(state)
    resumed = PaperTrader.from_state(config, state).replay(data, data.index[split])

    np.testing.assert_allclose(
        resumed["nav"].to_numpy(), uninterrupted["nav"].iloc[-len(resumed) :].to_numpy()
    )


def test_step_is_idempotent_per_bar() -> None:
    config = _config()
    data = _data(config)
    trader = PaperTrader(config)
    first = trader.step(data)
    again = trader.step(data)
    assert not first.skipped and again.skipped
    assert again.nav == first.nav


def test_state_of_another_experiment_is_refused(tmp_path: Path) -> None:
    config = _config()
    trader = PaperTrader(config)
    trader.step(_data(config))
    trader.save_state(tmp_path / "state.json")
    with pytest.raises(ValueError, match="belongs to experiment"):
        PaperTrader.from_state(_config(name="other"), tmp_path / "state.json")


def test_sync_over_missed_bars_compounds_like_bar_by_bar(constant_market: MarketData) -> None:
    engine = BacktestEngine(apply_funding=False)
    start = AccountState(nav=100.0, weights={"A": 0.5, "B": -0.25}, last_bar=None)
    start.last_bar = constant_market.index[2].isoformat()

    once = PaperBroker(engine, AccountState.from_dict(start.to_dict()))
    once.sync(constant_market.iloc(0, 7))

    stepwise = PaperBroker(engine, AccountState.from_dict(start.to_dict()))
    for stop in range(4, 8):
        stepwise.sync(constant_market.iloc(0, stop))

    assert once.nav == pytest.approx(stepwise.nav, rel=1e-12)
    pd.testing.assert_series_equal(once.weights, stepwise.weights)
    assert once.nav != 100.0


def test_execute_requires_a_synced_account(constant_market: MarketData) -> None:
    broker = PaperBroker(BacktestEngine())
    with pytest.raises(ValueError, match="sync"):
        broker.execute(pd.Series({"A": 1.0}), constant_market)


def test_strategies_that_learn_are_refused() -> None:
    with pytest.raises(ConfigurationError, match="fitted"):
        PaperTrader(_config(strategies=[{"name": "ml_directional"}]))


def test_cli_replay_reports_parity(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import yaml

    path = tmp_path / "paper.yaml"
    path.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    assert main(["paper", str(path), "--replay", "5"]) == 0
    assert "reproduces the backtest" in capsys.readouterr().out
    # A live step on synthetic data is refused with an actionable message.
    assert main(["paper", str(path), "--state-dir", str(tmp_path)]) == 2
