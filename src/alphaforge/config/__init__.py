"""Typed, validated experiment configuration (YAML/JSON -> pydantic models)."""

from alphaforge.config.loader import ConfigError, apply_overrides, config_to_dict, load_config
from alphaforge.config.schema import (
    BacktestConfig,
    CleaningConfig,
    CostConfig,
    DataConfig,
    ExperimentConfig,
    PortfolioConfig,
    ReportConfig,
    ResearchConfig,
    RiskConfig,
    StrategySpec,
    SyntheticConfig,
    WalkForwardConfig,
)

__all__ = [
    "BacktestConfig",
    "CleaningConfig",
    "ConfigError",
    "CostConfig",
    "DataConfig",
    "ExperimentConfig",
    "PortfolioConfig",
    "ReportConfig",
    "ResearchConfig",
    "RiskConfig",
    "StrategySpec",
    "SyntheticConfig",
    "WalkForwardConfig",
    "apply_overrides",
    "config_to_dict",
    "load_config",
]
