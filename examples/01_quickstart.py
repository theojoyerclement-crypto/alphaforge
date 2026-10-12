"""Quickstart: run the multi-strategy experiment from its config and inspect the results.

python examples/01_quickstart.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.config import load_config
from alphaforge.research import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_config(ROOT / "configs" / "multi_strategy.yaml")
    result = run_experiment(config)

    with pd.option_context(
        "display.float_format", "{:,.3f}".format, "display.width", 140, "display.max_columns", None
    ):
        print("\n=== Performance (net of fees, spread, impact, funding) ===")
        print(result.summary())
        print("\n=== Annualised PnL attribution ===")
        print(
            pd.DataFrame(
                {
                    k: v.pnl_attribution()
                    for k, v in {"portfolio": result.portfolio, **result.sleeves}.items()
                }
            )
        )
        print("\n=== Latest strategy allocation ===")
        print(result.targets.allocation.iloc[-1])

    if result.run_dir:
        print(f"\nFull HTML report: {result.run_dir / 'report.html'}")


if __name__ == "__main__":
    main()
