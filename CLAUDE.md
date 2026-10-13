# AlphaForge: notes for Claude

Crypto quant research framework. Package in `src/alphaforge/`, configs in `configs/`, tests in `tests/`.
Design rationale: `docs/ARCHITECTURE.md`; bias controls: `docs/METHODOLOGY.md`. Read those instead of re-deriving.

## Commands
- Tests: `python -m pytest -q` (~1 min; `pythonpath=src` is set in pyproject, no install needed)
- Single area: `python -m pytest tests/test_backtest.py -q`
- Run: `PYTHONPATH=src python -m alphaforge run configs/multi_strategy.yaml [--no-save] [--set a.b=v]`
- Alpha leaderboard: `PYTHONPATH=src python -m alphaforge research configs/multi_strategy.yaml`
- Paper parity demo: `PYTHONPATH=src python -m alphaforge paper configs/multi_strategy.yaml --replay 10`
- Everything a reviewer runs: `python scripts/smoke_test.py` (~4 min)
- Data audit: `PYTHONPATH=src python -m alphaforge validate configs/binance_perps_pit.yaml --set data.source=parquet`
- Static checks (same as CI): `ruff check src tests examples scripts`, `ruff format --check src tests examples scripts`, `mypy`
- Windows: use `python`, not `python3` (the latter is the Microsoft Store stub and prints nothing).

## Invariants (do not break)
- **Causality**: feature/strategy row `t` uses data ≤ `t` only. Forward returns appear only in `alpha/evaluation.py` and `ml/labeling.py` as targets.
  New features/strategies are auto-covered by the future-scrambling tests; keep them passing.
- **Strategy contract**: `generate_signals(data) -> DataFrame` in `[-1, 1]`, aligned to `data.close`, via `self.finalize(...)`.
  No sizing, vol targeting or costs inside strategies; those belong to `portfolio/` and `backtest/`.
- Strategies that learn set `requires_fit = True`; they only run under walk-forward (enforced in `research/experiment.py`).
- Panels are wide `time × symbol`, UTC tz-aware index. `NaN` close = untradable; never forward-fill across listing gaps.
- Engine identity `net = gross + funding − costs` is asserted in tests.
- `live/` (paper trading) must reuse the backtest's code: `mark_to_market`, `cost_model`, `cost_inputs`,
  `PortfolioConstructor`. `tests/test_live.py` asserts replayed paper NAV == `BacktestEngine` equity.
- Real-data README tables use `configs/binance_perps.yaml` and `configs/binance_perps_pit.yaml` with `--set data.end=2026-10-12`; data lives in `data/store/` (not committed).
- Datetime resolution varies in pandas 3 (us for synthetic, ms for Binance): never compare raw `asi8`/`.value` across indexes; convert with `.as_unit("ns")` or use Timedelta arithmetic.
- `load_market_data` keeps its `(data, cleaning)` API; `load_dataset` adds the quality audit and file provenance used by run manifests.
- Universe membership (`data.universe`) gates sizing in `PortfolioConstructor.tradable`; strategies still see every candidate.
- `backtest/walk_forward.py` imports strategies only under `TYPE_CHECKING` (avoids an import cycle; `tests/test_imports.py` guards it).

## Conventions
- Python ≥3.11, full type hints, Google-style docstrings explaining *why*. Line length 100 (ruff config in pyproject).
- pandas 3 is installed: use `panel_to_long` (not `stack`) to build long frames; `x / x.shift(1) - 1`, not `pct_change`.
- Register new pieces with `@register_feature` / `@register_strategy`; expose public names in the package `__init__`.
- Config changes: edit `config/schema.py` (pydantic, `extra="forbid"`) and keep `configs/*.yaml` valid (tested).
- Synthetic data is the test oracle: planted edges are trend, 1-bar reversal, funding carry. Don't recalibrate
  `SyntheticConfig` defaults without updating `tests/test_alpha.py` and the README results tables.
- Results quoted in the READMEs come from real runs of `configs/*.yaml`; re-run and update them if behaviour changes.
- `runs/` and `data/store/` are git-ignored artifacts.
