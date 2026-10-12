"""Alpha research: evaluate the alpha library, read the evidence, build a composite.

Walks through the workflow a researcher follows before an idea becomes a strategy:
IC and its decay, Newey-West t-stats, Benjamini-Hochberg across the whole library,
deflated Sharpe, then an IC-weighted composite of the survivors.

    python examples/02_alpha_research.py
"""

from __future__ import annotations

import pandas as pd

from alphaforge.alpha import AlphaResearcher, long_short_returns
from alphaforge.analytics import sharpe_ratio
from alphaforge.config import DataConfig, SyntheticConfig
from alphaforge.data import load_market_data

UNIVERSE = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT", "LTC", "ATOM"]


def main() -> None:
    data, _ = load_market_data(
        DataConfig(symbols=UNIVERSE, timeframe="1d", synthetic=SyntheticConfig(n_bars=2000)),
        seed=42,
    )
    researcher = AlphaResearcher(horizons=(1, 5, 10, 20), n_quantiles=3, fdr=0.10)
    report = researcher.evaluate(data)

    cols = [
        "category",
        "direction",
        "ic_mean",
        "ic_tstat_nw",
        "bh_significant",
        "rank_autocorr",
        "ls_sharpe",
        "deflated_sharpe",
        "score",
    ]
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 180):
        print("=== Alpha leaderboard ===")
        print(report.leaderboard[cols])
        print("\n=== IC decay (mean rank IC by horizon) ===")
        print(report.ic_decay().loc[report.leaderboard.index[:6]])

    survivors = (
        report.leaderboard.query("bh_significant and direction == 'long'").index[:4].tolist()
    )
    print(f"\nCombining BH survivors: {survivors}")
    composite = researcher.combine(data, survivors, ic_lookback=120)
    ls = long_short_returns(composite, data.returns()).dropna()
    singles = {
        name: sharpe_ratio(report.long_short_returns[name], data.periods_per_year)
        for name in survivors
    }
    print("Gross long-short Sharpe of each survivor:", {k: round(v, 2) for k, v in singles.items()})
    print(
        f"Gross long-short Sharpe of the composite: {sharpe_ratio(ls, data.periods_per_year):.2f}"
    )
    print("(Gross = before costs and execution delay; run a backtest before believing any of it.)")


if __name__ == "__main__":
    main()
