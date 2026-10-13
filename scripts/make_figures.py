"""Generate the README figures (docs/img/*.png, dpi 150) and the numbers they display.

    python scripts/make_figures.py

Needs the Binance data of docs/RESULTS.md ("Reproducing the real-data tables"); public market
data, no API key. Every number drawn on a figure is also written to docs/img/figures.json, and
tests/test_docs.py checks each one against docs/RESULTS.md, so a figure cannot show a value
the results do not publish.

Figures:
    equity_pit_vs_btc.png  point-in-time book (overlay v2) vs BTC buy & hold, with drawdowns
    cost_stress.png        the same reversal alpha under three execution assumptions
    sharpe_intervals.png   portfolio Sharpe with 95% block-bootstrap intervals
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alphaforge.analytics import max_drawdown, sharpe_interval, sharpe_ratio  # noqa: E402
from alphaforge.backtest import BacktestEngine  # noqa: E402
from alphaforge.config import load_config  # noqa: E402
from alphaforge.research import ExperimentResult, run_experiment  # noqa: E402

OUT = ROOT / "docs" / "img"
REAL = ["data.end=2026-10-12", "data.source=parquet"]
DPI = 150
# Reference palette, categorical slots 1-2 (validated pair) and chart chrome.
BOOK, BENCH = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def fmt(value: float, decimals: int = 2, signed: bool = False, pct: bool = False) -> str:
    """The exact string drawn on the figure (and checked against RESULTS.md)."""
    v = value * 100 if pct else value
    text = f"{v:+.{decimals}f}" if signed else f"{v:.{decimals}f}"
    return (text + "%") if pct else text


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _run(config: str, overrides: list[str]) -> ExperimentResult:
    return run_experiment(load_config(ROOT / "configs" / config, overrides), save=False)


def buy_and_hold(result: ExperimentResult, prefix: str = "BTC") -> pd.Series:
    """Net returns of holding one asset with the experiment's own engine and frictions."""
    data, cfg = result.data, result.config
    symbol = next(s for s in data.symbols if s.startswith(prefix))
    weights = pd.DataFrame(0.0, index=data.index, columns=data.symbols)
    weights[symbol] = 1.0
    engine = BacktestEngine.from_config(cfg.backtest, cfg.costs, None)
    return engine.run(weights, data, start=result.portfolio.returns.index[0]).returns


def equity_figure(book: pd.Series, bench: pd.Series, numbers: dict[str, str]) -> plt.Figure:
    """Growth of 1 and drawdown, book vs benchmark, on one shared time axis (no dual y)."""
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    fig.patch.set_facecolor(SURFACE)
    for series, color, key in ((book, BOOK, "book"), (bench, BENCH, "bench")):
        equity = (1 + series).cumprod()
        drawdown = equity / equity.cummax() - 1
        label = (
            f"{numbers[key + '_label']}  Sharpe {numbers[key + '_sharpe']}, "
            f"max DD {numbers[key + '_maxdd']}"
        )
        top.plot(equity.index, equity, color=color, linewidth=2, label=label)
        bottom.plot(drawdown.index, drawdown * 100, color=color, linewidth=1.5)
    _style(top)
    _style(bottom)
    top.set_yscale("log")
    top.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1.0, 2.0, 4.0]))
    top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    top.yaxis.set_minor_formatter(NullFormatter())
    top.set_ylabel("Growth of 1 (log)", color=INK_2, fontsize=9)
    bottom.set_ylabel("Drawdown (%)", color=INK_2, fontsize=9)
    top.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    top.set_title(
        "Point-in-time Binance book (overlay v2) vs BTC buy & hold, net of all costs",
        color=INK,
        fontsize=11,
        loc="left",
    )
    fig.tight_layout()
    return fig


def cost_stress_figure(labels: list[str], values: list[float], shown: list[str]) -> plt.Figure:
    """Sharpe of one alpha as frictions are added: three levels, each labelled with its value."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    fig.patch.set_facecolor(SURFACE)
    positions = range(len(values))
    ax.bar(positions, values, width=0.55, color=BOOK)
    for x, (v, text) in enumerate(zip(values, shown, strict=True)):
        ax.annotate(
            text,
            (x, v),
            xytext=(0, 6 if v >= 0 else -14),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=10,
        )
    for x in range(1, len(values)):  # connectors show the step from one assumption to the next
        ax.plot([x - 0.725, x - 0.275], [values[x - 1]] * 2, color=INK_2, linewidth=1, ls=":")
    ax.axhline(0, color=INK_2, linewidth=1)
    pad = 0.15 * (max(values) - min(values))  # keep value labels clear of title and ticks
    ax.set_ylim(min(0.0, min(values)) - pad, max(0.0, max(values)) + pad)
    ax.set_xticks(list(positions), labels, fontsize=9, color=INK)
    ax.set_ylabel("Annualised Sharpe", color=INK_2, fontsize=9)
    _style(ax)
    ax.set_title(
        "A t = 5.5 alpha is not a strategy: same signal, more realistic execution",
        color=INK,
        fontsize=11,
        loc="left",
    )
    fig.tight_layout()
    return fig


def interval_figure(rows: list[dict[str, str | float]]) -> plt.Figure:
    """Forest plot: point Sharpe and 95% interval per book, with the zero line."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    fig.patch.set_facecolor(SURFACE)
    for y, row in enumerate(reversed(rows)):
        color = BENCH if row["benchmark"] else BOOK
        ax.plot([row["low"], row["high"]], [y, y], color=color, linewidth=2)
        ax.plot([row["estimate"]], [y], "o", color=color, markersize=8)
        ax.annotate(
            f"{row['estimate_txt']}  [{row['low_txt']}, {row['high_txt']}]",
            (row["high"], y),
            xytext=(8, -3),
            textcoords="offset points",
            color=INK_2,
            fontsize=9,
        )
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(
        range(len(rows)), [str(r["label"]) for r in reversed(rows)], color=INK, fontsize=9
    )
    ax.set_xlabel("Annualised Sharpe, 95% block-bootstrap interval", color=INK_2, fontsize=9)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=ax.get_xlim()[1] + 1.2)  # room for the value labels
    ax.set_title(
        "Only the synthetic book, where edges were planted, clears zero",
        color=INK,
        fontsize=11,
        loc="left",
    )
    fig.tight_layout()
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    numbers: dict[str, object] = {}

    # (a) point-in-time book vs BTC
    pit = _run("binance_perps_pit.yaml", REAL)
    book, bench = pit.portfolio.returns, buy_and_hold(pit)
    ppy = pit.data.periods_per_year
    eq = {
        "book_label": "PIT book",
        "book_sharpe": fmt(sharpe_ratio(book, ppy)),
        "book_maxdd": fmt(max_drawdown(book), 1, pct=True),
        "bench_label": "BTC buy & hold",
        "bench_sharpe": fmt(sharpe_ratio(bench, ppy)),
        "bench_maxdd": fmt(max_drawdown(bench), 1, pct=True),
    }
    equity_figure(book, bench, eq).savefig(OUT / "equity_pit_vs_btc.png", dpi=DPI)
    numbers["equity_pit_vs_btc"] = {k: v for k, v in eq.items() if not k.endswith("_label")}

    # (b) cost stress, with the three override sets documented in configs/cost_stress.yaml
    steps = [
        (
            "Frictionless,\nsame-bar",
            [
                "backtest.execution_delay=0",
                "costs.model=linear",
                "costs.fee_bps=0",
                "costs.half_spread_bps=0",
            ],
        ),
        ("+ fees, spread,\nsqrt impact", ["backtest.execution_delay=0"]),
        ("+ 1 bar latency\n(default)", []),
    ]
    values = [_run("cost_stress.yaml", ov).portfolio.metrics().sharpe for _, ov in steps]
    shown = [fmt(v, signed=True) for v in values]
    cost_stress_figure([s[0] for s in steps], values, shown).savefig(
        OUT / "cost_stress.png", dpi=DPI
    )
    numbers["cost_stress"] = shown

    # (c) Sharpe intervals (same method and seed as examples/05_sharpe_confidence.py)
    books = [
        ("Synthetic multi-strategy", _run("multi_strategy.yaml", []).portfolio.returns, False),
        ("Walk-forward ML book", _run("ml_walk_forward.yaml", []).portfolio.returns, False),
        ("Binance, fixed universe", _run("binance_perps.yaml", REAL).portfolio.returns, False),
        ("Binance, point-in-time", book, False),
        ("BTC buy & hold", bench, True),
    ]
    rows = []
    for label, returns, is_bench in books:
        ci = sharpe_interval(returns, ppy)
        rows.append(
            {
                "label": label,
                "benchmark": is_bench,
                "estimate": ci.estimate,
                "low": ci.low,
                "high": ci.high,
                "estimate_txt": fmt(ci.estimate),
                "low_txt": fmt(ci.low),
                "high_txt": fmt(ci.high),
            }
        )
    interval_figure(rows).savefig(OUT / "sharpe_intervals.png", dpi=DPI)
    numbers["sharpe_intervals"] = [[r["estimate_txt"], r["low_txt"], r["high_txt"]] for r in rows]

    (OUT / "figures.json").write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(list(OUT.glob('*.png')))} figures and figures.json to {OUT}")


if __name__ == "__main__":
    main()
