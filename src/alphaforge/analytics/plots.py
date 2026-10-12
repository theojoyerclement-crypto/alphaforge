"""Static research charts (matplotlib, backend-independent ``Figure`` objects)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from alphaforge.analytics.metrics import (
    drawdown_series,
    equity_curve,
    monthly_returns,
    rolling_sharpe,
)

_PALETTE = ["#1f4e79", "#c0504d", "#4f8a3c", "#8064a2", "#d18f1e", "#2c9faf", "#7f7f7f", "#b85c9e"]


def _style(ax: object) -> None:
    ax.grid(alpha=0.25, linewidth=0.6)  # type: ignore[attr-defined]
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)  # type: ignore[attr-defined]


def plot_equity_curves(
    returns: Mapping[str, pd.Series], title: str = "Equity curves", log: bool = True
) -> Figure:
    """Compounded equity of several return streams (the first is drawn thicker)."""
    fig = Figure(figsize=(11, 4.8))
    ax = fig.subplots()
    for i, (name, r) in enumerate(returns.items()):
        ax.plot(
            equity_curve(r),
            label=name,
            color=_PALETTE[i % len(_PALETTE)],
            lw=2.2 if i == 0 else 1.1,
        )
    if log:
        ax.set_yscale("log")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel("Growth of 1")
    ax.legend(frameon=False, ncol=min(4, len(returns)), fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_underwater(returns: pd.Series, title: str = "Drawdown") -> Figure:
    fig = Figure(figsize=(11, 2.8))
    ax = fig.subplots()
    dd = drawdown_series(returns)
    ax.fill_between(dd.index, dd.to_numpy(), 0.0, color="#c0504d", alpha=0.45, lw=0)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_rolling_sharpe(returns: pd.Series, window: int, periods_per_year: float) -> Figure:
    fig = Figure(figsize=(11, 2.8))
    ax = fig.subplots()
    ax.plot(rolling_sharpe(returns, window, periods_per_year), color="#1f4e79", lw=1.2)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title(f"Rolling Sharpe ({window} bars)", loc="left", fontweight="bold")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_monthly_heatmap(returns: pd.Series) -> Figure:
    table = monthly_returns(returns)
    fig = Figure(figsize=(11, 0.45 * len(table) + 1.5))
    ax = fig.subplots()
    values = table.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0.1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    ax.imshow(values, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns])
    ax.set_yticks(range(table.shape[0]), [str(i) for i in table.index])
    for (i, j), v in np.ndenumerate(values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=7)
    ax.set_title("Monthly returns", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(returns: pd.DataFrame, title: str = "Strategy correlation") -> Figure:
    corr = returns.corr()
    n = len(corr)
    fig = Figure(figsize=(1.0 * n + 3, 0.8 * n + 2))
    ax = fig.subplots()
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), corr.index, fontsize=8)
    for (i, j), v in np.ndenumerate(corr.to_numpy()):
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if abs(v) > 0.6 else "black",
        )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_ic_decay(ic_decay: pd.DataFrame, top: int = 8) -> Figure:
    """Mean IC versus forward horizon for the strongest alphas."""
    order = ic_decay.abs().max(axis=1).sort_values(ascending=False).index[:top]
    fig = Figure(figsize=(9, 4))
    ax = fig.subplots()
    for i, name in enumerate(order):
        ax.plot(
            ic_decay.columns,
            ic_decay.loc[name],
            marker="o",
            label=name,
            color=_PALETTE[i % len(_PALETTE)],
        )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Forward horizon (bars)")
    ax.set_ylabel("Mean rank IC")
    ax.set_title("IC decay", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_allocation(allocation: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(11, 3))
    ax = fig.subplots()
    ax.stackplot(
        allocation.index,
        allocation.T.to_numpy(),
        labels=allocation.columns,
        colors=_PALETTE[: allocation.shape[1]],
        alpha=0.85,
    )
    ax.set_ylim(0, 1)
    ax.set_title("Strategy allocation", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=min(5, allocation.shape[1]), loc="upper left")
    _style(ax)
    fig.tight_layout()
    return fig
