"""One-command check for a reviewer: test suite, a reference backtest, and a paper replay.

    python scripts/smoke_test.py            # ~3 min, offline, same on Windows/macOS/Linux
    python scripts/smoke_test.py --quick    # skip the test suite

Works from a fresh clone without ``pip install -e .`` (it puts ``src/`` on the path) as long
as the requirements are installed. Exits non-zero on the first failure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="skip the pytest suite")
    args = parser.parse_args()
    started = time.perf_counter()

    if not args.quick:
        _banner("1/3  Test suite (causality, accounting identities, CV leakage, end-to-end)")
        code = subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
        if code != 0:
            return code

    from alphaforge.cli import main as cli

    _banner("2/3  Reference backtest on the synthetic market (planted edges, net of costs)")
    code = cli(["run", str(ROOT / "configs" / "multi_strategy.yaml")])
    if code != 0:
        return code

    _banner("3/3  Paper-trading replay: live code path must reproduce the backtest")
    code = cli(["paper", str(ROOT / "configs" / "multi_strategy.yaml"), "--replay", "10"])
    if code != 0:
        return code

    runs = sorted((ROOT / "runs").glob("*_multi_strategy"))
    report = runs[-1] / "report.html" if runs else None
    _banner(f"All checks passed in {time.perf_counter() - started:.0f}s")
    if report is not None:
        print(f"Open the HTML report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
