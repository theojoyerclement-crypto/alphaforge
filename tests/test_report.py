"""HTML report: formatting of counts, drawdowns and dates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphaforge.analytics.report import _fmt, format_drawdowns

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_counts_are_integers_and_ratios_are_percent() -> None:
    assert _fmt("n_periods", 1819.0) == "1,819"
    assert _fmt("max_drawdown_duration", 1350) == "1,350"
    assert _fmt("max_drawdown", -0.18204) == "-18.20%"
    assert _fmt("sharpe", 1.367) == "1.37"


def test_drawdown_table_display() -> None:
    utc = pd.Timestamp("2020-05-22", tz="UTC")
    dd = pd.DataFrame(
        {
            "peak": [utc, utc],
            "trough": [utc + pd.Timedelta(days=83), utc + pd.Timedelta(days=10)],
            "recovery": [utc + pd.Timedelta(days=386), pd.NaT],
            "depth": [-0.18204671683700546, -0.05],
            "bars_to_trough": [83, 10],
            "bars_to_recovery": [303.0, np.nan],
            "total_bars": [386, 40],
        }
    )
    shown = format_drawdowns(dd)
    assert shown.loc[0, "peak"] == "2020-05-22"
    assert shown.loc[0, "depth"] == "-18.20%"
    assert shown.loc[0, "bars_to_recovery"] == "303"
    assert shown.loc[1, "recovery"] == "not recovered"
    assert shown.loc[1, "bars_to_recovery"] == "n/a"
    html = shown.to_html()
    assert "00:00:00" not in html and "+00:00" not in html


def _report_html(tmp_path, config: dict) -> str:  # type: ignore[no-untyped-def]
    from alphaforge.analytics.report import generate_report
    from alphaforge.backtest import BacktestEngine

    idx = pd.date_range("2022-01-01", periods=120, tz="UTC")
    close = pd.DataFrame({"A": 100 * np.exp(np.cumsum(np.full(120, 0.001)))}, index=idx)
    from alphaforge.data.market_data import MarketData

    frame = close.rename(columns={"A": "close"}).assign(
        open=close["A"], high=close["A"], low=close["A"], volume=1e9
    )
    data = MarketData.from_frames({"A": frame}, "1d")
    result = BacktestEngine().run(pd.DataFrame(1.0, index=idx, columns=["A"]), data)
    out = generate_report(result, tmp_path, "t", config=config, plots=False)
    return (out / "report.html").read_text(encoding="utf-8")


def test_header_states_the_data_source(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from alphaforge.analytics.report import SYNTHETIC_BANNER
    from alphaforge.config import config_to_dict, load_config

    synthetic = config_to_dict(load_config(CONFIGS / "multi_strategy.yaml"))
    html = _report_html(tmp_path / "s", synthetic)
    assert SYNTHETIC_BANNER in html
    assert "Data source: synthetic market (seeded, planted edges), 12 assets" in html
    assert html.index(SYNTHETIC_BANNER) < html.index("<h1>"), "banner sits above the title"

    pit = config_to_dict(load_config(CONFIGS / "binance_perps_pit.yaml"))
    html = _report_html(tmp_path / "p", pit)
    assert SYNTHETIC_BANNER not in html
    assert (
        "Data source: Binance perpetual futures market data, 1d bars, point-in-time universe: "
        "top 12 of 29 candidates" in html
    )

    fixed = config_to_dict(load_config(CONFIGS / "binance_perps.yaml"))
    assert "fixed universe of 12 symbols" in _report_html(tmp_path / "f", fixed)
    assert "Data source: not recorded" in _report_html(tmp_path / "n", {})
    summary = (tmp_path / "s" / "summary.md").read_text(encoding="utf-8")
    assert SYNTHETIC_BANNER in summary


def embedded_charts(html: str) -> list[tuple[str, float]]:
    """``(alt text, pixel std)`` of every embedded PNG; a blank chart has std ~ 0."""
    import base64
    import io
    import re

    import matplotlib.image as mpimg

    charts = []
    for alt, payload in re.findall(
        r"<img alt='([^']*)' src='data:image/png;base64,([^']+)'>", html
    ):
        pixels = mpimg.imread(io.BytesIO(base64.b64decode(payload)), format="png")
        charts.append((alt, float(np.asarray(pixels)[..., :3].std())))
    return charts


def test_full_report_embeds_six_non_blank_charts(tmp_path: Path) -> None:
    from alphaforge.config import ExperimentConfig
    from alphaforge.research import run_experiment

    config = ExperimentConfig.model_validate(
        {
            "name": "charts",
            "data": {"symbols": ["A", "B", "C", "D"], "synthetic": {"n_bars": 400}},
            "strategies": [{"name": "ts_momentum"}, {"name": "funding_carry"}],
            "portfolio": {"allocation_min_history": 20, "allocation_lookback": 40},
            "report": {"save": True, "output_dir": str(tmp_path)},
        }
    )
    run_dir = run_experiment(config).run_dir
    assert run_dir is not None
    charts = embedded_charts((run_dir / "report.html").read_text(encoding="utf-8"))
    names = [alt for alt, _ in charts]
    assert names == ["Equity", "Drawdown", "Rolling Sharpe", "Monthly", "Correlation", "Allocation"]
    assert all(std > 0.02 for _, std in charts), charts
