"""HTML report: formatting of counts, drawdowns and dates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.analytics.report import _fmt, format_drawdowns


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
