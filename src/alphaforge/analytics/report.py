"""Self-contained research reports: HTML (charts embedded), Markdown summary, CSV/JSON."""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from alphaforge.analytics.drawdown import drawdown_table
from alphaforge.analytics.metrics import monthly_returns
from alphaforge.analytics.plots import (
    plot_allocation,
    plot_correlation_heatmap,
    plot_equity_curves,
    plot_monthly_heatmap,
    plot_rolling_sharpe,
    plot_underwater,
)
from alphaforge.backtest.results import BacktestResult

_PERCENT = {
    "total_return",
    "cagr",
    "volatility",
    "max_drawdown",
    "hit_rate",
    "var_95",
    "cvar_95",
    "annual_cost_drag",
}

_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;
  background:#f6f7f9;color:#1d2430}
main{max-width:1100px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:17px;margin:34px 0 10px;border-bottom:1px solid #d9dde3;padding-bottom:6px}
.meta{color:#5b6573;font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;
  margin-top:18px}
.kpi{background:#fff;border:1px solid #e1e4e8;border-radius:8px;padding:12px 14px}
.kpi .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
.kpi .l{font-size:12px;color:#5b6573}
table{border-collapse:collapse;font-size:12.5px;background:#fff;width:100%;
  font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid #eceef1;padding:5px 9px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#f0f2f5}
.wrap{overflow-x:auto}
img{width:100%;background:#fff;border:1px solid #e1e4e8;border-radius:6px;margin:6px 0}
.note{font-size:12.5px;color:#5b6573;background:#fff;border-left:3px solid #1f4e79;
  padding:8px 12px}
"""


def _fig_to_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fmt(key: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if key in _PERCENT:
        return f"{value:.2%}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _section(heading: str, body: str) -> str:
    return f'<h2>{heading}</h2><div class="wrap">{body}</div>'


def metrics_table(results: Mapping[str, BacktestResult]) -> pd.DataFrame:
    """Side-by-side metrics (rows: metrics, columns: results)."""
    return pd.DataFrame({name: r.metrics().to_series() for name, r in results.items()})


def generate_report(
    portfolio: BacktestResult,
    output_dir: str | Path,
    title: str,
    sleeves: Mapping[str, BacktestResult] | None = None,
    allocation: pd.DataFrame | None = None,
    extra_tables: Mapping[str, pd.DataFrame] | None = None,
    config: Mapping[str, Any] | None = None,
    plots: bool = True,
    html: bool = True,
) -> Path:
    """Write the full report for a backtest into ``output_dir`` and return the directory.

    Files: ``metrics.csv``, ``metrics.json``, ``drawdowns.csv``, ``monthly_returns.csv``,
    ``pnl_attribution.csv``, per-result time series, ``summary.md`` and ``report.html``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sleeves = dict(sleeves or {})
    everything = {portfolio.name: portfolio, **sleeves}

    table = metrics_table(everything)
    table.to_csv(out / "metrics.csv")
    (out / "metrics.json").write_text(
        json.dumps(
            {k: r.metrics().to_dict() for k, r in everything.items()}, indent=2, default=str
        ),
        encoding="utf-8",
    )
    dd = drawdown_table(portfolio.returns)
    dd.to_csv(out / "drawdowns.csv", index=False)
    monthly_returns(portfolio.returns).to_csv(out / "monthly_returns.csv")
    attribution = pd.DataFrame({k: r.pnl_attribution() for k, r in everything.items()})
    attribution.to_csv(out / "pnl_attribution.csv")
    for result in everything.values():
        result.save(out / "timeseries")
    for name, extra in (extra_tables or {}).items():
        extra.to_csv(out / f"{name}.csv")

    pm = portfolio.metrics()
    summary = [
        f"# {title}",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC}. "
        f"Period {portfolio.returns.index[0].date()} -> {portfolio.returns.index[-1].date()} "
        f"({pm.n_periods} bars).",
        "",
        "| Metric | " + " | ".join(table.columns) + " |",
        "|---|" + "---|" * len(table.columns),
    ]
    for metric, row in table.iterrows():
        summary.append(f"| {metric} | " + " | ".join(_fmt(str(metric), v) for v in row) + " |")
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    if not html:
        return out

    images: list[tuple[str, str]] = []
    if plots:
        curves = {portfolio.name: portfolio.returns, **{k: v.returns for k, v in sleeves.items()}}
        ppy = portfolio.periods_per_year
        images.append(
            (
                "Equity",
                _fig_to_base64(plot_equity_curves(curves, "Portfolio and sleeves (net of costs)")),
            )
        )
        images.append(("Drawdown", _fig_to_base64(plot_underwater(portfolio.returns))))
        images.append(
            (
                "Rolling Sharpe",
                _fig_to_base64(plot_rolling_sharpe(portfolio.returns, int(ppy / 2), ppy)),
            )
        )
        images.append(("Monthly", _fig_to_base64(plot_monthly_heatmap(portfolio.returns))))
        if len(sleeves) > 1:
            corr = pd.DataFrame({k: v.returns for k, v in sleeves.items()})
            images.append(("Correlation", _fig_to_base64(plot_correlation_heatmap(corr))))
        if allocation is not None and allocation.shape[1] > 1:
            images.append(
                (
                    "Allocation",
                    _fig_to_base64(plot_allocation(allocation.loc[portfolio.returns.index[0] :])),
                )
            )

    kpis = [
        ("CAGR", "cagr"),
        ("Sharpe", "sharpe"),
        ("Sortino", "sortino"),
        ("Calmar", "calmar"),
        ("Max drawdown", "max_drawdown"),
        ("Volatility", "volatility"),
        ("Profit factor", "profit_factor"),
        ("Cost drag / yr", "annual_cost_drag"),
    ]
    pm_dict = pm.to_dict()
    kpi_html = "".join(
        f'<div class="kpi"><div class="v">{_fmt(key, pm_dict[key])}</div>'
        f'<div class="l">{label}</div></div>'
        for label, key in kpis
    )
    formatted = table.copy().astype(object)
    for metric in formatted.index:
        formatted.loc[metric] = [_fmt(str(metric), v) for v in table.loc[metric]]
    period = f"{portfolio.returns.index[0].date()} → {portfolio.returns.index[-1].date()}"
    sections = [
        f"<h1>{title}</h1>",
        f'<div class="meta">{period} · {pm.n_periods} bars · '
        "net of fees, spread, impact and funding</div>",
        f'<div class="kpis">{kpi_html}</div>',
    ]
    for label, img in images:
        sections.append(f"<h2>{label}</h2><img alt='{label}' src='data:image/png;base64,{img}'>")
    sections.append(_section("Metrics", formatted.to_html(border=0)))
    pct = attribution.map(lambda v: f"{v:.2%}")
    sections.append(_section("PnL attribution (annualised)", pct.to_html(border=0)))
    if not dd.empty:
        sections.append(_section("Worst drawdowns", dd.to_html(border=0, index=False)))
    for name, extra in (extra_tables or {}).items():
        body = extra.to_html(border=0, float_format=lambda v: f"{v:,.4f}")
        sections.append(_section(name.replace("_", " ").title(), body))
    if config is not None:
        dumped = json.dumps(config, indent=2, default=str)
        sections.append(f"<h2>Configuration</h2><pre class='note'>{dumped}</pre>")
    page = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
        f"<style>{_CSS}</style></head><body><main>{''.join(sections)}</main></body></html>"
    )
    (out / "report.html").write_text(page, encoding="utf-8")
    return out
