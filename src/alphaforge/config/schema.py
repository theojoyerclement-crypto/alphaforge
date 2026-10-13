"""Experiment configuration schema.

Every tunable of an experiment lives here, validated by pydantic. Unknown keys are rejected
(``extra="forbid"``) so a typo in a YAML file fails loudly instead of silently using a default.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from alphaforge.utils.time import timeframe_to_freq


def _date_to_str(value: object) -> object:
    # YAML turns an unquoted 2021-01-01 (in a file or a --set override) into a date object.
    return value.isoformat() if isinstance(value, date) else value


DateStr = Annotated[str, BeforeValidator(_date_to_str)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SyntheticConfig(_Model):
    """Parameters of the synthetic crypto market generator (tests, demos, method validation)."""

    n_bars: int = Field(1500, ge=100)
    start: DateStr = "2020-01-01"
    annual_vol: float = Field(0.65, gt=0)
    annual_drift: float = 0.05
    factor_share: float = Field(0.55, ge=0, le=1, description="Variance share of the market factor")
    trend_persistence: float = Field(0.97, ge=0, lt=1, description="AR(1) of the latent drift")
    trend_strength: float = Field(0.04, ge=0, description="Latent drift vol, in units of bar vol")
    carry_strength: float = Field(0.4, ge=0, description="Price penalty for crowded funding")
    reversal_strength: float = Field(0.04, ge=0, description="1-bar idiosyncratic reversal")
    tail_dof: float = Field(4.0, gt=2, description="Student-t degrees of freedom")
    stagger_listings: bool = True


class DataConfig(_Model):
    """Where market data comes from and which universe/timeframe to load."""

    source: Literal["synthetic", "ccxt", "parquet"] = "synthetic"
    exchange: str = "binance"
    market_type: Literal["spot", "swap"] = "swap"
    symbols: list[str] = Field(min_length=1)
    timeframe: str = "1d"
    start: DateStr | None = None
    end: DateStr | None = None
    storage_dir: str = "data/store"
    include_funding: bool = True
    use_cache: bool = True
    synthetic: SyntheticConfig = SyntheticConfig()

    @field_validator("timeframe")
    @classmethod
    def _check_timeframe(cls, value: str) -> str:
        timeframe_to_freq(value)
        return value


class CleaningConfig(_Model):
    """Data cleaning policy."""

    max_ffill_bars: int = Field(3, ge=0, description="Longest gap repaired by forward fill")
    outlier_mad_threshold: float = Field(20.0, gt=0, description="Spike threshold in MADs")
    repair_outliers: bool = True


class StrategySpec(_Model):
    """One strategy sleeve: registry name, unique id and constructor parameters."""

    name: str
    id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.id or self.name


class PortfolioConfig(_Model):
    """Portfolio construction: per-sleeve sizing, cross-strategy allocation and limits."""

    vol_halflife: int = Field(20, ge=2, description="EWMA half-life (bars) for risk estimates")
    sleeve_target_vol: float | None = Field(0.15, gt=0)
    allocation: Literal["equal", "inverse_vol", "risk_parity"] = "risk_parity"
    allocation_lookback: int = Field(120, ge=20)
    allocation_min_history: int = Field(60, ge=10)
    rebalance_every: int = Field(5, ge=1, description="Strategy-allocation rebalance period")
    correlation_threshold: float | None = Field(0.80, gt=0, le=1)
    target_vol: float | None = Field(0.20, gt=0, description="Annualised portfolio target vol")
    max_gross_leverage: float = Field(2.0, gt=0)
    max_net_leverage: float = Field(1.0, gt=0)
    max_position: float = Field(0.30, gt=0, description="Max |weight| per asset")


class RiskConfig(_Model):
    """Path-dependent risk overlays applied inside the backtest loop."""

    max_drawdown: float | None = Field(0.30, gt=0, lt=1, description="Hard stop: go flat")
    drawdown_soft_limit: float | None = Field(0.15, gt=0, lt=1, description="Start de-risking")
    cooldown_bars: int = Field(20, ge=0, description="Bars flat after a hard-stop breach")

    @model_validator(mode="after")
    def _soft_below_hard(self) -> RiskConfig:
        if (
            self.max_drawdown is not None
            and self.drawdown_soft_limit is not None
            and self.drawdown_soft_limit >= self.max_drawdown
        ):
            raise ValueError("drawdown_soft_limit must be below max_drawdown")
        return self


class CostConfig(_Model):
    """Transaction cost model."""

    model: Literal["linear", "sqrt_impact"] = "sqrt_impact"
    fee_bps: float = Field(4.0, ge=0, description="Exchange fee per unit traded (taker)")
    half_spread_bps: float = Field(1.5, ge=0)
    impact_coefficient: float = Field(0.10, ge=0, description="Square-root impact constant")
    adv_lookback: int = Field(20, ge=1, description="Bars used for average dollar volume")
    vol_halflife: int = Field(20, ge=2)


class BacktestConfig(_Model):
    """Execution and accounting conventions."""

    initial_capital: float = Field(1_000_000.0, gt=0)
    execution_delay: int = Field(
        1, ge=0, description="Bars between signal (close t) and execution (close t+delay)"
    )
    apply_funding: bool = True
    start: DateStr | None = Field(None, description="Evaluation start (warm-up excluded)")


class WalkForwardConfig(_Model):
    """Walk-forward (out-of-sample) evaluation; mandatory for strategies that fit."""

    enabled: bool = False
    train_bars: int = Field(500, ge=50)
    test_bars: int = Field(120, ge=10)
    expanding: bool = True
    embargo_bars: int = Field(5, ge=0)


class ResearchConfig(_Model):
    """Alpha research settings."""

    horizons: list[int] = Field(default_factory=lambda: [1, 5, 10, 20])
    n_quantiles: int = Field(3, ge=2)
    fdr: float = Field(0.10, gt=0, lt=1, description="Benjamini-Hochberg false discovery rate")
    top_k: int = Field(5, ge=1)

    @field_validator("horizons")
    @classmethod
    def _positive(cls, value: list[int]) -> list[int]:
        if not value or any(h < 1 for h in value):
            raise ValueError("horizons must be a non-empty list of positive integers")
        return sorted(set(value))


class ReportConfig(_Model):
    """Output settings."""

    output_dir: str = "runs"
    save: bool = True
    plots: bool = True
    html: bool = True


class ExperimentConfig(_Model):
    """Root configuration of an AlphaForge experiment."""

    name: str
    description: str = ""
    seed: int = 42
    log_level: str = "INFO"
    data: DataConfig
    cleaning: CleaningConfig = CleaningConfig()
    strategies: list[StrategySpec] = Field(min_length=1)
    portfolio: PortfolioConfig = PortfolioConfig()
    risk: RiskConfig = RiskConfig()
    costs: CostConfig = CostConfig()
    backtest: BacktestConfig = BacktestConfig()
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    research: ResearchConfig = ResearchConfig()
    report: ReportConfig = ReportConfig()

    @model_validator(mode="after")
    def _unique_labels(self) -> ExperimentConfig:
        labels = [spec.label for spec in self.strategies]
        duplicates = {label for label in labels if labels.count(label) > 1}
        if duplicates:
            raise ValueError(f"Duplicate strategy labels {sorted(duplicates)}; set a unique `id`")
        return self
