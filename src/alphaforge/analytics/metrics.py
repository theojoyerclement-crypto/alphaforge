"""Performance metrics on per-period return series.

Conventions: arithmetic Sharpe/Sortino annualised by ``sqrt(periods_per_year)``; CAGR is
geometric; drawdowns are measured on the compounded equity curve; risk-free rate is zero
(crypto has no canonical risk-free asset; subtract one upstream if needed).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


def _clean(returns: pd.Series) -> pd.Series:
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compounded equity from per-period returns."""
    return initial * (1.0 + returns.fillna(0.0)).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown from the running peak (``<= 0``)."""
    eq = equity_curve(returns)
    peak = np.maximum(eq.cummax(), 1.0)
    return (eq / peak - 1.0).rename("drawdown")


def cagr(returns: pd.Series, periods_per_year: float) -> float:
    r = _clean(returns)
    if r.empty:
        return float("nan")
    growth = float((1.0 + r).prod())
    years = len(r) / periods_per_year
    return growth ** (1.0 / years) - 1.0 if growth > 0 and years > 0 else -1.0


def annualized_volatility(returns: pd.Series, periods_per_year: float) -> float:
    r = _clean(returns)
    return float(r.std() * np.sqrt(periods_per_year)) if len(r) > 1 else float("nan")


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    r = _clean(returns)
    std = r.std()
    return (
        float(r.mean() / std * np.sqrt(periods_per_year))
        if len(r) > 1 and std > 0
        else float("nan")
    )


def sortino_ratio(returns: pd.Series, periods_per_year: float) -> float:
    """Mean return over downside deviation (target 0)."""
    r = _clean(returns)
    downside = np.sqrt((np.minimum(r, 0.0) ** 2).mean())
    return float(r.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    """Most negative peak-to-trough drawdown (a negative number)."""
    r = _clean(returns)
    return float(drawdown_series(r).min()) if len(r) else float("nan")


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest stretch, in bars, spent below a previous peak."""
    dd = drawdown_series(_clean(returns)).to_numpy()
    longest = current = 0
    for underwater in dd < 0:
        current = current + 1 if underwater else 0
        longest = max(longest, current)
    return longest


def calmar_ratio(returns: pd.Series, periods_per_year: float) -> float:
    mdd = max_drawdown(returns)
    return cagr(returns, periods_per_year) / abs(mdd) if mdd < 0 else float("nan")


def profit_factor(returns: pd.Series) -> float:
    """Gross profits over gross losses."""
    r = _clean(returns)
    losses = -r[r < 0].sum()
    return float(r[r > 0].sum() / losses) if losses > 0 else float("nan")


def hit_rate(returns: pd.Series) -> float:
    """Fraction of non-zero periods that were profitable."""
    r = _clean(returns)
    active = r[r != 0]
    return float((active > 0).mean()) if len(active) else float("nan")


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Historical one-period VaR (a negative return)."""
    r = _clean(returns)
    return float(np.quantile(r, 1.0 - level)) if len(r) else float("nan")


def conditional_value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall: mean return beyond the VaR."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    var = np.quantile(r, 1.0 - level)
    return float(r[r <= var].mean())


def tail_ratio(returns: pd.Series) -> float:
    """95th percentile gain over |5th percentile| loss."""
    r = _clean(returns)
    lo = abs(np.quantile(r, 0.05)) if len(r) else 0.0
    return float(np.quantile(r, 0.95) / lo) if lo > 0 else float("nan")


def rolling_sharpe(returns: pd.Series, window: int, periods_per_year: float) -> pd.Series:
    roll = returns.rolling(window, min_periods=window // 2)
    return (roll.mean() / roll.std() * np.sqrt(periods_per_year)).rename("rolling_sharpe")


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Compounded returns by calendar month (rows: years, columns: months 1-12)."""
    r = returns.fillna(0.0)
    index = pd.DatetimeIndex(r.index)
    grouped = (1.0 + r).groupby([index.year, index.month]).prod() - 1.0
    table = grouped.unstack()
    table.index.name, table.columns.name = "year", "month"
    return table


@dataclass(frozen=True)
class PerformanceMetrics:
    """Standard performance report of a return stream."""

    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration: int
    profit_factor: float
    hit_rate: float
    skew: float
    kurtosis: float
    var_95: float
    cvar_95: float
    tail_ratio: float
    avg_turnover: float | None
    annual_turnover: float | None
    avg_gross_leverage: float | None
    annual_cost_drag: float | None
    n_periods: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)

    def to_series(self, name: str = "metrics") -> pd.Series:
        return pd.Series(self.to_dict(), name=name)


def compute_metrics(
    returns: pd.Series,
    periods_per_year: float,
    turnover: pd.Series | None = None,
    gross_leverage: pd.Series | None = None,
    costs: pd.Series | None = None,
) -> PerformanceMetrics:
    """Compute :class:`PerformanceMetrics` for a per-period return series."""
    r = _clean(returns)
    nan = float("nan")
    return PerformanceMetrics(
        total_return=float((1.0 + r).prod() - 1.0) if len(r) else nan,
        cagr=cagr(r, periods_per_year),
        volatility=annualized_volatility(r, periods_per_year),
        sharpe=sharpe_ratio(r, periods_per_year),
        sortino=sortino_ratio(r, periods_per_year),
        calmar=calmar_ratio(r, periods_per_year),
        max_drawdown=max_drawdown(r),
        max_drawdown_duration=max_drawdown_duration(r),
        profit_factor=profit_factor(r),
        hit_rate=hit_rate(r),
        skew=float(stats.skew(r)) if len(r) > 2 else nan,
        kurtosis=float(stats.kurtosis(r)) if len(r) > 3 else nan,
        var_95=value_at_risk(r),
        cvar_95=conditional_value_at_risk(r),
        tail_ratio=tail_ratio(r),
        avg_turnover=float(turnover.mean()) if turnover is not None else None,
        annual_turnover=float(turnover.mean() * periods_per_year) if turnover is not None else None,
        avg_gross_leverage=float(gross_leverage.mean()) if gross_leverage is not None else None,
        annual_cost_drag=float(costs.mean() * periods_per_year) if costs is not None else None,
        n_periods=len(r),
    )
