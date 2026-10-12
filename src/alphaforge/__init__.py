"""AlphaForge: a quantitative alpha research framework for crypto markets.

Layers (each importable on its own):

- :mod:`alphaforge.data`        ingestion, cleaning, storage, the ``MarketData`` panel container
- :mod:`alphaforge.features`    causal feature engine (technical, microstructure, carry)
- :mod:`alphaforge.alpha`       alpha library, IC analysis, multiple-testing aware validation
- :mod:`alphaforge.strategies`  strategy families (momentum, mean reversion, regime, carry, ML)
- :mod:`alphaforge.portfolio`   sizing, volatility targeting, risk limits, strategy allocation
- :mod:`alphaforge.backtest`    event-loop engine, cost models, walk-forward, purged CV
- :mod:`alphaforge.ml`          labeling, regime detection, directional models, meta-labeling
- :mod:`alphaforge.analytics`   performance metrics, drawdowns, plots, reports
- :mod:`alphaforge.research`    config-driven experiment orchestration
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
