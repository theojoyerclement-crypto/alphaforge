"""Strategy layer (importing this package registers all built-in strategies)."""

from alphaforge.strategies.base import (
    Strategy,
    StrategyFactory,
    available_strategies,
    build_strategy,
    register_strategy,
)
from alphaforge.strategies.carry import FundingCarry
from alphaforge.strategies.composite import AlphaComposite
from alphaforge.strategies.mean_reversion import ShortTermReversal, ZScoreMeanReversion
from alphaforge.strategies.ml_based import MetaLabeledStrategy, MLDirectional, RegimeAdaptive
from alphaforge.strategies.momentum import CrossSectionalMomentum, TimeSeriesMomentum
from alphaforge.strategies.regime import RegimeSwitching

__all__ = [
    "AlphaComposite",
    "CrossSectionalMomentum",
    "FundingCarry",
    "MLDirectional",
    "MetaLabeledStrategy",
    "RegimeAdaptive",
    "RegimeSwitching",
    "ShortTermReversal",
    "Strategy",
    "StrategyFactory",
    "TimeSeriesMomentum",
    "ZScoreMeanReversion",
    "available_strategies",
    "build_strategy",
    "register_strategy",
]
