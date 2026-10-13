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
