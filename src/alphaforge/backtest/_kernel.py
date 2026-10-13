"""Compiled backtest loop (numba), a faster twin of the Python loop in ``engine.py``.

The Python loop in :meth:`BacktestEngine._run_python` is the reference implementation: it is
what the tests, the paper trader and the documentation describe. This kernel re-implements the
same arithmetic for the built-in cost models and drawdown controller only, and
``tests/test_kernel.py`` asserts that both produce the same results to 1e-10. Profiling showed
the Python loop's time goes to per-bar call overhead on tiny arrays (``docs/PERFORMANCE.md``);
compiling the whole loop removes that overhead without changing what is computed.

numba is optional (``pip install alphaforge[fast]``). Without it, :data:`AVAILABLE` is False and
the engine uses the Python loop.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised in environments with and without numba
    from numba import njit

    AVAILABLE = True
except ImportError:  # pragma: no cover
    AVAILABLE = False

LINEAR, SQRT_IMPACT = 0, 1


def _cost(
    trades: np.ndarray,
    nav: float,
    vol: np.ndarray,
    adv: np.ndarray,
    kind: int,
    linear_rate: float,
    impact_k: float,
) -> float:
    """Total cost as a fraction of NAV; mirrors LinearCostModel / SquareRootImpactModel."""
    total = 0.0
    for j in range(trades.shape[0]):
        traded = abs(trades[j])
        cost = traded * linear_rate
        if kind == SQRT_IMPACT:
            a = adv[j]
            participation = traded * nav / a if (math.isfinite(a) and a > 0.0) else 1.0
            sigma = vol[j]
            if math.isnan(sigma):
                sigma = 0.0
            cost += traded * (impact_k * sigma * math.sqrt(participation))
        total += cost
    return total


def _controller_update(
    equity: float,
    state: np.ndarray,
    max_dd: float,
    soft: float,
    cooldown_bars: int,
) -> float:
    """DrawdownController.update with its state in ``state = [hwm, cooldown, breaches]``.

    ``max_dd`` / ``soft`` are NaN when the corresponding limit is disabled; ``hwm`` is NaN
    before the first update.
    """
    if math.isnan(state[0]):
        state[0] = equity
    if state[1] > 0:
        state[1] -= 1
        if state[1] == 0:
            state[0] = equity
        return 0.0
    state[0] = max(state[0], equity)
    dd = 1.0 - equity / state[0]
    if not math.isnan(max_dd) and dd >= max_dd:
        state[2] += 1
        state[1] = cooldown_bars
        if state[1] == 0:
            state[0] = equity
        return 0.0
    if not math.isnan(soft) and dd > soft:
        if math.isnan(max_dd):
            return 0.5
        return max(0.0, 1.0 - (dd - soft) / (max_dd - soft))
    return 1.0


def _loop(
    desired: np.ndarray,
    R: np.ndarray,
    F: np.ndarray,
    vol: np.ndarray,
    adv: np.ndarray,
    first: int,
    initial_capital: float,
    kind: int,
    linear_rate: float,
    impact_k: float,
    use_controller: bool,
    use_shadow: bool,
    max_dd: float,
    soft: float,
    cooldown_bars: int,
    state: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """The bar loop of ``BacktestEngine._run_python``, on plain arrays."""
    n_total, n_assets = R.shape
    n_bars = n_total - first
    net = np.zeros(n_bars)
    gross = np.zeros(n_bars)
    funding = np.zeros(n_bars)
    costs = np.zeros(n_bars)
    turnover = np.zeros(n_bars)
    risk_scale = np.ones(n_bars)
    positions = np.zeros((n_bars, n_assets))
    equity = np.zeros(n_bars)
    shadow_equity = np.zeros(n_bars)
    wiped = -1  # first bar index (in evaluation bars) where the NAV was wiped out

    nav = initial_capital
    w = np.zeros(n_assets)
    nav_s = initial_capital
    w_s = np.zeros(n_assets)
    drift = np.empty(n_assets)
    drift_s = np.empty(n_assets)
    target = np.empty(n_assets)
    trades = np.empty(n_assets)
    for i in range(n_bars):
        t = first + i
        # mark to market (engine.mark_to_market)
        price_pnl = 0.0
        funding_pnl = 0.0
        for j in range(n_assets):
            price_pnl += w[j] * R[t, j]
            funding_pnl -= w[j] * F[t, j]
        growth = 1.0 + price_pnl + funding_pnl
        if growth <= 0.0:
            drift[:] = 0.0
            nav_pre = 0.0
            if wiped < 0:
                wiped = i
        else:
            for j in range(n_assets):
                drift[j] = w[j] * (1.0 + R[t, j]) / growth
            nav_pre = nav * growth

        basis = nav_pre
        if use_shadow:
            growth_s = 1.0
            for j in range(n_assets):
                growth_s += w_s[j] * R[t, j] - w_s[j] * F[t, j]
            if growth_s > 0.0:
                for j in range(n_assets):
                    drift_s[j] = w_s[j] * (1.0 + R[t, j]) / growth_s
                basis = nav_s * growth_s
            else:
                drift_s[:] = 0.0
                basis = 0.0
            if basis > 0.0:
                for j in range(n_assets):
                    trades[j] = desired[t, j] - drift_s[j]
                cost_s = _cost(trades, basis, vol[t], adv[t], kind, linear_rate, impact_k)
                nav_s = basis * (1.0 - cost_s)
                w_s[:] = desired[t]
            else:
                nav_s = 0.0
                w_s[:] = 0.0
            shadow_equity[i] = nav_s

        if not use_controller or nav_pre <= 0.0:
            scale = 1.0
        elif basis > 0.0:
            scale = _controller_update(basis, state, max_dd, soft, cooldown_bars)
        else:
            scale = 0.0

        if nav_pre > 0.0:
            for j in range(n_assets):
                target[j] = desired[t, j] * scale
                trades[j] = target[j] - drift[j]
            cost = _cost(trades, nav_pre, vol[t], adv[t], kind, linear_rate, impact_k)
            nav_post = nav_pre * (1.0 - cost)
        else:
            target[:] = 0.0
            trades[:] = 0.0
            cost = 0.0
            nav_post = 0.0

        net[i] = nav_post / nav - 1.0 if nav > 0.0 else 0.0
        gross[i] = price_pnl
        funding[i] = funding_pnl
        costs[i] = cost * growth
        total_traded = 0.0
        for j in range(n_assets):
            total_traded += abs(trades[j])
        turnover[i] = total_traded
        risk_scale[i] = scale
        positions[i] = target
        equity[i] = nav_post
        nav = nav_post
        w[:] = target
    wiped_arr = np.array([wiped], dtype=np.int64)
    return (
        net, gross, funding, costs, turnover, risk_scale, positions, equity, shadow_equity,
        wiped_arr,
    )  # fmt: skip


if AVAILABLE:
    _cost = njit(cache=True)(_cost)
    _controller_update = njit(cache=True)(_controller_update)
    run_loop: Any = njit(cache=True)(_loop)
else:  # pragma: no cover
    run_loop = None
