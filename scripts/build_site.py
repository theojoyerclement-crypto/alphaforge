"""Build the GitHub Pages site: a landing page and two HTML reports.

    python scripts/build_site.py --binance   # locally: refresh docs/reports/binance_perps_pit
    python scripts/build_site.py --out site  # CI: rebuild the synthetic report, assemble site/

The point-in-time Binance report is the reference result. CI does not rebuild it: the data is
not stored in the repository and is not fetched from CI (Binance restricts access from several
jurisdictions), so it is generated locally
from public market data with the same pinned window as docs/RESULTS.md and committed under
docs/reports/. The synthetic report is rebuilt on every deployment. Numbers on the landing page
are read from each run's metrics.json, never typed.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alphaforge.analytics.report import SYNTHETIC_BANNER, describe_data_source  # noqa: E402
from alphaforge.config import load_config  # noqa: E402
from alphaforge.research import run_experiment  # noqa: E402

COMMITTED = ROOT / "docs" / "reports"
BINANCE = ("binance_perps_pit", ["data.end=2026-10-12", "data.source=parquet"])
SYNTHETIC = ("multi_strategy", [])
KEEP = ("report.html", "manifest.json", "metrics.json", "config.json")


def run_report(name: str, overrides: list[str], dest: Path) -> Path:
    """Run a reference config with a report and copy its shareable files to ``dest``."""
    config = load_config(ROOT / "configs" / f"{name}.yaml", overrides)
    result = run_experiment(config, save=True)
    assert result.run_dir is not None
    dest.mkdir(parents=True, exist_ok=True)
    for file in KEEP:
        shutil.copy2(result.run_dir / file, dest / file)
    return dest


def _headline(report_dir: Path) -> dict[str, str]:
    """Portfolio CAGR, Sharpe and max drawdown as published (1 dp %, 2 dp Sharpe)."""
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))["portfolio"]
    return {
        "CAGR": f"{metrics['cagr']:.1%}",
        "Sharpe": f"{metrics['sharpe']:.2f}",
        "Max drawdown": f"{metrics['max_drawdown']:.1%}",
    }


def _card(href: str, title: str, report_dir: Path, lead: str, banner: bool) -> str:
    config = json.loads((report_dir / "config.json").read_text(encoding="utf-8"))
    source, _ = describe_data_source(config)
    stats = "".join(
        f'<div class="kpi"><div class="v">{v}</div><div class="l">{k}</div></div>'
        for k, v in _headline(report_dir).items()
    )
    flag = f'<div class="banner">{SYNTHETIC_BANNER}</div>' if banner else ""
    return (
        f'<section class="card">{flag}<h2><a href="{href}">{html.escape(title)}</a></h2>'
        f'<p class="lead">{lead}</p><p class="source">{html.escape(source)}</p>'
        f'<div class="kpis">{stats}</div>'
        f'<p><a class="open" href="{href}">Open the full report →</a></p></section>'
    )


def landing_page(binance_dir: Path, synthetic_dir: Path) -> str:
    """The site's index.html: the real-data reference first, the synthetic test second."""
    css = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#f6f7f9;
  color:#1d2430}
main{max-width:900px;margin:0 auto;padding:32px 20px 60px}
h1{font-size:26px;margin:0 0 6px} h2{font-size:19px;margin:6px 0}
a{color:#1f5fae} .muted{color:#5b6573;font-size:14px}
.card{background:#fff;border:1px solid #e1e4e8;border-radius:10px;padding:18px 20px;margin:18px 0}
.card.ref{border:2px solid #2a78d6}
.tag{display:inline-block;font-size:12px;font-weight:600;color:#fff;background:#2a78d6;
  border-radius:4px;padding:2px 8px}
.lead{margin:6px 0} .source{color:#5b6573;font-size:13px;margin:4px 0 10px}
.kpis{display:flex;gap:10px;flex-wrap:wrap}
.kpi{border:1px solid #e1e4e8;border-radius:8px;padding:8px 12px;min-width:110px}
.kpi .v{font-size:18px;font-weight:600;font-variant-numeric:tabular-nums}
.kpi .l{font-size:12px;color:#5b6573}
.banner{background:#fff4e5;border:1px solid #eb6834;border-left:6px solid #eb6834;
  border-radius:6px;padding:8px 12px;margin-bottom:10px;font-weight:600}
.open{font-weight:600}
"""
    reference = _card(
        "binance-pit/",
        "Binance perpetuals, point-in-time universe (reference result)",
        binance_dir,
        "Real market data, net of fees, spread, square-root impact and funding, with the "
        "pre-registered drawdown overlay (v2). This is the number to judge the strategies by.",
        banner=False,
    ).replace(
        '<section class="card">', '<section class="card ref"><span class="tag">Reference</span>'
    )
    synthetic = _card(
        "synthetic/",
        "Synthetic multi-strategy book",
        synthetic_dir,
        "A seeded simulated market with three planted edges. It checks that the research stack "
        "finds what was planted; its returns say nothing about real markets.",
        banner=True,
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>AlphaForge reports</title><style>{css}</style></head><body><main>"
        "<h1>AlphaForge: example reports</h1>"
        '<p class="muted">Two runs of the reference configs, as produced by '
        "<code>alphaforge run</code>. Full tables, confidence intervals and caveats: "
        '<a href="https://github.com/theojoyerclement-crypto/alphaforge/blob/main/docs/RESULTS.md">'
        "docs/RESULTS.md</a>. Source: "
        '<a href="https://github.com/theojoyerclement-crypto/alphaforge">GitHub</a>.</p>'
        f"{reference}{synthetic}"
        '<p class="muted">The Binance report is generated locally from public market data '
        "(<code>python scripts/build_site.py --binance</code>) and committed: the data is not "
        "stored in the repository and is not fetched from CI. The synthetic report is rebuilt "
        "on every deployment.</p></main></body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binance", action="store_true", help="refresh the committed report")
    parser.add_argument("--out", type=Path, default=ROOT / "site")
    args = parser.parse_args()

    binance_dir = COMMITTED / BINANCE[0]
    if args.binance:
        run_report(*BINANCE, binance_dir)
        print(f"Refreshed {binance_dir}")
        return

    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(binance_dir, out / "binance-pit")
    synthetic_dir = run_report(*SYNTHETIC, out / "synthetic")
    (out / "index.html").write_text(landing_page(out / "binance-pit", synthetic_dir), "utf-8")
    print(f"Site written to {out}")


if __name__ == "__main__":
    main()
