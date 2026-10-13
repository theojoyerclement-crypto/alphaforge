"""Cross-strategy capital allocation with correlation management."""

from __future__ import annotations

import numpy as np
import pandas as pd


def risk_parity_weights(cov: np.ndarray, max_iter: int = 100, tol: float = 1e-12) -> np.ndarray:
    """Equal-risk-contribution weights (long-only, sum to 1).

    Unlike inverse-vol, ERC accounts for correlations: two highly correlated sleeves share one
    risk budget. Assets with zero variance get zero weight.

    Solved as the strictly convex problem ``min ½ xᵀΣx − (1/n) Σ log xᵢ`` (Spinu, 2013), whose
    unique minimiser normalised to sum 1 is the ERC portfolio. Damped Newton steps converge
    quadratically (typically < 10 iterations); the earlier multiplicative fixed point needed
    hundreds on correlated sleeves and stopped at a looser optimum (weights differ < 1e-9).

    Args:
        tol: Stop when every scaled gradient ``|xᵢ (Σx)ᵢ − 1/n|`` is below this, i.e. risk
            contributions are equal to ``tol`` relative to the total.
    """
    var = np.diag(cov).copy()
    active = var > 1e-16
    w = np.zeros(len(var))
    if not active.any():
        return w
    sub = cov[np.ix_(active, active)]
    budget = 1.0 / len(sub)
    x = 1.0 / np.sqrt(np.diag(sub))
    x /= np.sqrt(x @ sub @ x)  # unit portfolio variance: the optimum has xᵀΣx = 1
    for _ in range(max_iter):
        grad = sub @ x - budget / x
        if np.abs(grad * x).max() < tol:
            break
        step = np.linalg.solve(sub + np.diag(budget / x**2), grad)
        scale = 1.0
        while np.any(x - scale * step <= 0):  # stay in the positive orthant
            scale *= 0.5
        x = x - scale * step
    w[active] = x / x.sum()
    return w


def inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    """Weights proportional to ``1 / sigma`` (zero-variance sleeves get zero)."""
    vol = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    inv = np.where(vol > 1e-8, 1.0 / np.where(vol > 1e-8, vol, 1.0), 0.0)
    return inv / inv.sum() if inv.sum() > 0 else inv


def correlation_penalty(
    weights: np.ndarray, returns: np.ndarray, threshold: float, haircut: float = 0.5
) -> np.ndarray:
    """Haircut the weaker sleeve of every pair whose correlation exceeds ``threshold``.

    "Weaker" means the lower trailing Sharpe. Prevents two near-duplicate strategies from
    quietly doubling the same bet.
    """
    std = returns.std(axis=0)
    ok = std > 0
    if ok.sum() < 2:
        return weights
    corr = np.corrcoef(returns[:, ok], rowvar=False)
    idx = np.flatnonzero(ok)
    sharpe = np.where(std > 0, returns.mean(axis=0) / np.where(std > 0, std, 1.0), -np.inf)
    w = weights.copy()
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if corr[a, b] > threshold:
                weaker = idx[a] if sharpe[idx[a]] < sharpe[idx[b]] else idx[b]
                w[weaker] *= haircut
    return w / w.sum() if w.sum() > 0 else w


class StrategyAllocator:
    """Time-varying, causal allocation of risk across strategy sleeves.

    At each rebalance bar ``t`` the weights are estimated from sleeve returns up to ``t``
    (sleeve returns at ``t`` are realised at the close of ``t``) and held until the next
    rebalance. Equal weights are used until ``min_history`` bars are available.

    Args:
        method: ``"equal"``, ``"inverse_vol"`` or ``"risk_parity"`` (ERC).
        lookback: Estimation window in bars.
        rebalance_every: Rebalance period in bars.
        correlation_threshold: Pairwise correlation above which the weaker sleeve is cut.
        min_history: Bars before the first estimated allocation.
    """

    def __init__(
        self,
        method: str = "risk_parity",
        lookback: int = 120,
        rebalance_every: int = 5,
        correlation_threshold: float | None = 0.8,
        min_history: int = 60,
    ) -> None:
        if method not in {"equal", "inverse_vol", "risk_parity"}:
            raise ValueError(f"Unknown allocation method {method!r}")
        self.method = method
        self.lookback = lookback
        self.rebalance_every = rebalance_every
        self.correlation_threshold = correlation_threshold
        self.min_history = min_history

    def _estimate(self, window: np.ndarray) -> np.ndarray:
        n_sleeves = window.shape[1]
        if self.method == "equal":
            w = np.full(n_sleeves, 1.0 / n_sleeves)
        else:
            cov = np.cov(window, rowvar=False) if n_sleeves > 1 else np.atleast_2d(window.var())
            w = (
                risk_parity_weights(cov)
                if self.method == "risk_parity"
                else inverse_vol_weights(cov)
            )
        if self.correlation_threshold is not None and n_sleeves > 1:
            w = correlation_penalty(w, window, self.correlation_threshold)
        return w

    def allocate(self, sleeve_returns: pd.DataFrame) -> pd.DataFrame:
        """Allocation weights (rows sum to 1) aligned with ``sleeve_returns``."""
        R = sleeve_returns.fillna(0.0).to_numpy()
        n, s = R.shape
        out = np.empty((n, s))
        current = np.full(s, 1.0 / s)
        # first bar with any sleeve activity: history before it is meaningless
        active_rows = np.flatnonzero(np.abs(R).sum(axis=1) > 0)
        first_active = int(active_rows[0]) if active_rows.size else n
        for t in range(n):
            history = t + 1 - first_active
            if history >= self.min_history and (t - first_active) % self.rebalance_every == 0:
                window = R[max(first_active, t + 1 - self.lookback) : t + 1]
                estimate = self._estimate(window)
                if estimate.sum() > 0:
                    current = estimate
            out[t] = current
        return pd.DataFrame(out, index=sleeve_returns.index, columns=sleeve_returns.columns)
