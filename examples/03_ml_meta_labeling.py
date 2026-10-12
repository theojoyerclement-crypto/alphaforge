"""ML layer: purged cross-validation, meta-labeling and regime detection.

python examples/03_ml_meta_labeling.py
"""

from __future__ import annotations

import pandas as pd

from alphaforge.backtest import BacktestEngine, SquareRootImpactModel
from alphaforge.config import DataConfig, SyntheticConfig
from alphaforge.data import load_market_data
from alphaforge.ml import (
    DirectionalModel,
    GaussianMixtureRegimeModel,
    MetaLabeler,
    market_regime_features,
)
from alphaforge.portfolio import PortfolioConstructor
from alphaforge.strategies import build_strategy

UNIVERSE = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT"]


def main() -> None:
    data, _ = load_market_data(
        DataConfig(symbols=UNIVERSE, synthetic=SyntheticConfig(n_bars=1800)), seed=11
    )
    split = 1200
    train, test_start = data.iloc(0, split), data.index[split]

    # 1) Purged K-fold of pooled direction models: honest, leakage-free scores.
    print("=== Purged 5-fold CV, 5-bar relative direction (mean over folds) ===")
    rows = {}
    for name in ("logistic", "lightgbm"):
        cv = DirectionalModel(name, horizon=5).cross_validate(train, n_splits=5, embargo_pct=0.02)
        rows[name] = cv[["accuracy", "auc", "log_loss"]].mean()
    print(pd.DataFrame(rows).T.round(4))
    print("AUC ~0.5 (or slightly below) = no learnable edge at this horizon. Under a null signal,")
    print("K-fold is biased *below* chance: the fit on the other blocks is anti-correlated with")
    print("the held-out block. Read it as 'no edge', never as 'invert the model'.")

    # 2) Meta-labeling a trend-following primary, trained on the first 1200 bars only.
    primary = build_strategy("ts_momentum").generate_signals(data)
    meta = MetaLabeler(
        "random_forest", pt_mult=1.5, sl_mult=1.0, max_holding=10, min_probability=0.52
    )
    meta.fit(train, primary.iloc[:split])
    sized = meta.size_positions(data, primary)

    constructor = PortfolioConstructor()
    engine = BacktestEngine(SquareRootImpactModel(), execution_delay=1)
    raw = engine.run(constructor.size_sleeve(primary, data), data, start=test_start, name="primary")
    filt = engine.run(
        constructor.size_sleeve(sized, data), data, start=test_start, name="meta_labeled"
    )
    table = pd.DataFrame({r.name: r.metrics().to_series() for r in (raw, filt)})
    print("\n=== Out-of-sample: primary vs meta-labeled (net of costs) ===")
    print(
        table.loc[
            ["cagr", "volatility", "sharpe", "max_drawdown", "hit_rate", "annual_turnover"]
        ].round(3)
    )

    # 3) Regimes learned on the training window, applied out of sample.
    feats = market_regime_features(data)
    gmm = GaussianMixtureRegimeModel(n_regimes=3).fit(feats.iloc[:split])
    regimes = gmm.predict(feats)
    summary = (
        feats.assign(regime=regimes)
        .iloc[split:]
        .groupby("regime")
        .agg(
            bars=("volatility", "size"), volatility=("volatility", "mean"), trend=("trend", "mean")
        )
    )
    print("\n=== Out-of-sample regime profile (0 = calmest) ===")
    print(summary.round(3))


if __name__ == "__main__":
    main()
