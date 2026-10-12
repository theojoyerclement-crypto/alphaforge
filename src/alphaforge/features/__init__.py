"""Feature engineering engine (importing this package registers all built-in features)."""

from alphaforge.features import carry, microstructure, technical  # noqa: F401 - registration
from alphaforge.features.base import (
    FeatureDefinition,
    FeatureEngine,
    FeatureRequest,
    compute_feature,
    feature_label,
    get_feature,
    list_features,
    register_feature,
)
from alphaforge.features.transforms import (
    cs_demean,
    cs_rank,
    cs_zscore,
    long_short_quantiles,
    normalize_gross,
    panel_to_long,
    ts_zscore,
)

__all__ = [
    "FeatureDefinition",
    "FeatureEngine",
    "FeatureRequest",
    "compute_feature",
    "cs_demean",
    "cs_rank",
    "cs_zscore",
    "feature_label",
    "get_feature",
    "list_features",
    "long_short_quantiles",
    "normalize_gross",
    "panel_to_long",
    "register_feature",
    "ts_zscore",
]
