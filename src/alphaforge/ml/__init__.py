"""Machine learning layer: labeling, regime detection, directional models, meta-labeling."""

from alphaforge.ml.labeling import ewma_volatility, fixed_horizon_labels, triple_barrier_labels
from alphaforge.ml.meta_labeling import MetaLabeler, probability_to_size
from alphaforge.ml.models import DEFAULT_ML_FEATURES, DirectionalModel, make_classifier
from alphaforge.ml.regime import (
    GaussianMixtureRegimeModel,
    VolatilityRegimeClassifier,
    market_regime_features,
)

__all__ = [
    "DEFAULT_ML_FEATURES",
    "DirectionalModel",
    "GaussianMixtureRegimeModel",
    "MetaLabeler",
    "VolatilityRegimeClassifier",
    "ewma_volatility",
    "fixed_horizon_labels",
    "make_classifier",
    "market_regime_features",
    "probability_to_size",
    "triple_barrier_labels",
]
