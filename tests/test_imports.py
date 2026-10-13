"""Every subpackage must import cleanly on its own (catches circular imports that a
particular test ordering would hide)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGES = [
    "alphaforge",
    "alphaforge.config",
    "alphaforge.data",
    "alphaforge.features",
    "alphaforge.alpha",
    "alphaforge.strategies",
    "alphaforge.portfolio",
    "alphaforge.backtest",
    "alphaforge.ml",
    "alphaforge.analytics",
    "alphaforge.analytics.report",
    "alphaforge.research",
    "alphaforge.live",
    "alphaforge.cli",
]


@pytest.mark.parametrize("module", PACKAGES)
def test_import_in_fresh_interpreter(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        env={**os.environ, "PYTHONPATH": str(SRC)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
