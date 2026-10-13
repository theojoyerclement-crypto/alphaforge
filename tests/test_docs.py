"""Docs consistency: a README may only quote numbers that docs/RESULTS.md publishes.

docs/RESULTS.md is the single home of published results. The READMEs summarise it, so any
decimal or percentage in their prose must appear there verbatim (after normalising the French
decimal comma and the Unicode minus). A typo or a stale figure in a README fails CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "RESULTS.md"

_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+\.\d+(?:\s?%)?")
# Numbers that are not results: tool versions and test tolerances quoted in prose.
_NOT_RESULTS = {"3.11", "3.12", "3.13", "3.14"}


def _prose(markdown: str) -> str:
    """Drop fenced code, image/badge lines and link targets: only prose numbers are checked."""
    text = re.sub(r"```.*?```", "", markdown, flags=re.S)
    text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith(("![", "[!["))
    )
    return re.sub(r"\]\([^)]*\)", "]", text)


def _numbers(markdown: str, french: bool = False) -> set[str]:
    # Unicode minus (U+2212), narrow and regular no-break spaces (U+202F, U+00A0).
    text = _prose(markdown).replace(chr(0x2212), "-")
    text = text.replace(chr(0x202F), " ").replace(chr(0xA0), " ")
    if french:  # 0,52 -> 0.52 ; [-0,05 ; 1,30] -> [-0.05 ; 1.30]
        text = re.sub(r"(\d),(\d)", r"\1.\2", text)
    found = set()
    for token in _NUMBER.findall(text):
        value = token.replace(" ", "").lstrip("+")
        if value not in _NOT_RESULTS:
            found.add(value)
    return found


def test_results_document_exists_and_has_the_reference_sections() -> None:
    text = RESULTS.read_text(encoding="utf-8")
    for heading in (
        "### 5. Real market: Binance USDT-M perpetuals",
        "### Drawdown overlay v1 vs v2",
        "### How much of this is noise?",
        "## Reproducing the real-data tables",
    ):
        assert heading in text


@pytest.mark.parametrize(("readme", "french"), [("README.md", False), ("README.fr.md", True)])
def test_readme_numbers_are_published_in_results(readme: str, french: bool) -> None:
    published = _numbers(RESULTS.read_text(encoding="utf-8"))
    quoted = _numbers((ROOT / readme).read_text(encoding="utf-8"), french=french)
    missing = sorted(quoted - published)
    assert not missing, f"{readme} quotes numbers absent from docs/RESULTS.md: {missing}"


def _load_make_figures():  # type: ignore[no-untyped-def]
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_figures", ROOT / "scripts" / "make_figures.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_figure_numbers_are_published_in_results() -> None:
    """Every value drawn on a README figure (docs/img/figures.json) is a published number."""
    import json

    shown = json.loads((ROOT / "docs" / "img" / "figures.json").read_text(encoding="utf-8"))
    flat: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        else:
            flat.append(str(node))

    walk(shown)
    published = _numbers(RESULTS.read_text(encoding="utf-8"))
    missing = sorted(v.lstrip("+") for v in flat if v.lstrip("+") not in published)
    assert flat and not missing, f"figures show numbers absent from docs/RESULTS.md: {missing}"
    for name in ("equity_pit_vs_btc", "cost_stress", "sharpe_intervals"):
        assert (ROOT / "docs" / "img" / f"{name}.png").stat().st_size > 10_000


def test_figure_functions_render_without_exchange_data(tmp_path: Path) -> None:
    import numpy as np
    import pandas as pd

    mf = _load_make_figures()
    idx = pd.date_range("2022-01-01", periods=300, tz="UTC")
    rng = np.random.default_rng(0)
    book, bench = (pd.Series(rng.normal(0, 0.02, 300), index=idx) for _ in range(2))
    labels = {"book_label": "a", "bench_label": "b"}
    numbers = {
        **labels,
        **{f"{k}_{m}": "0.00" for k in ("book", "bench") for m in ("sharpe", "maxdd")},
    }
    rows = [
        {
            "label": "x",
            "benchmark": False,
            "estimate": 0.5,
            "low": -0.2,
            "high": 1.1,
            "estimate_txt": "0.50",
            "low_txt": "-0.20",
            "high_txt": "1.10",
        },
    ]
    for i, fig in enumerate(
        (
            mf.equity_figure(book, bench, numbers),
            mf.cost_stress_figure(["a", "b", "c"], [1.8, -0.3, -2.9], ["+1.80", "-0.30", "-2.90"]),
            mf.interval_figure(rows),
        )
    ):
        fig.savefig(tmp_path / f"{i}.png", dpi=mf.DPI)
        assert (tmp_path / f"{i}.png").stat().st_size > 5_000
    assert mf.fmt(-0.0835) == "-0.08" and mf.fmt(1.83, signed=True) == "+1.83"
    assert mf.fmt(-0.4271, 1, pct=True) == "-42.7%"


def test_pages_landing_quotes_published_numbers_and_flags_synthetic(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    assert spec is not None and spec.loader is not None
    site = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(site)

    built = site.build(tmp_path / "site")
    page = (built / "index.html").read_text(encoding="utf-8")

    # Every relative link must resolve in the built site (a folder link needs index.html).
    import re

    for href in re.findall(r'href="([^"]+)"', page):
        if href.startswith("http"):
            continue
        target = built / href
        assert (target / "index.html" if href.endswith("/") else target).is_file(), href

    assert page.index("reference result") < page.index("Synthetic multi-strategy book")
    synthetic_card = page[page.index("Synthetic multi-strategy book") - 400 :]
    assert site.SYNTHETIC_BANNER in synthetic_card
    assert site.SYNTHETIC_BANNER not in page[: page.index("Synthetic multi-strategy book") - 400]
    missing = sorted(_numbers(page) - _numbers(RESULTS.read_text(encoding="utf-8")))
    assert not missing, f"landing page shows numbers absent from docs/RESULTS.md: {missing}"
