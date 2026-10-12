"""Alpha research: candidate generation, IC analysis, multiple-testing aware validation."""

from alphaforge.alpha.base import Alpha
from alphaforge.alpha.evaluation import (
    AlphaEvaluation,
    cross_sectional_ic,
    evaluate_alpha,
    forward_returns,
    long_short_returns,
    newey_west_tstat,
    quantile_returns,
    rank_autocorrelation,
)
from alphaforge.alpha.library import default_alpha_library, get_alpha
from alphaforge.alpha.research import AlphaResearcher, AlphaResearchReport
from alphaforge.alpha.validation import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
)

__all__ = [
    "Alpha",
    "AlphaEvaluation",
    "AlphaResearchReport",
    "AlphaResearcher",
    "benjamini_hochberg",
    "cross_sectional_ic",
    "default_alpha_library",
    "deflated_sharpe_ratio",
    "evaluate_alpha",
    "expected_max_sharpe",
    "forward_returns",
    "get_alpha",
    "long_short_returns",
    "minimum_track_record_length",
    "newey_west_tstat",
    "probabilistic_sharpe_ratio",
    "quantile_returns",
    "rank_autocorrelation",
]
