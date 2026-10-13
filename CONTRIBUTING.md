# Contributing

## Run the checks (same as CI)

```bash
pip install -e ".[all]"
python -m pytest -q                         # whole suite, ~2 min
python -m pytest tests/test_backtest.py -q  # one area
ruff check src tests examples scripts && ruff format --check src tests examples scripts
mypy
```

## Add a strategy

1. Subclass `Strategy` in `src/alphaforge/strategies/`, decorate it with `@register_strategy`,
   and give it a unique `name` and a `family`.
2. Implement `generate_signals(data) -> DataFrame` in `[-1, 1]`, aligned to `data.close`, and
   return it through `self.finalize(...)`. Row `t` may only use data up to `t`.
3. Don't size positions, target volatility or charge costs inside a strategy. Those belong to
   `portfolio/` and `backtest/`.
4. If it learns parameters, set `requires_fit = True` and implement `fit`. It will then only
   run under walk-forward.
5. Expose it in `strategies/__init__.py` and use it from YAML: `- {name: my_strategy, params: {...}}`.

The causality (future-scrambling) and bounds tests in `tests/test_strategies.py` pick up every
registered strategy automatically, so they must keep passing. Features work the same way
with `@register_feature`.

## Ground rules

- Do not recalibrate `SyntheticConfig` defaults without updating `tests/test_alpha.py` and the
  README results tables: the synthetic market is the test oracle.
- Numbers quoted in the READMEs come from real runs of `configs/*.yaml`. If behaviour changes,
  re-run them and update the tables, including unflattering ones.
- Never commit API keys, `data/store/`, `runs/` or `paper/`.
