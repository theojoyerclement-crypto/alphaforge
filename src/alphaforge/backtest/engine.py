"""Event-loop backtest engine.

Timeline of one bar ``t`` (all at the close of ``t``):

1. **Mark to market.** Positions held since close ``t-1`` earn the bar's return
   ``r_t`` and pay/receive the bar's funding ``f_t`` (longs pay positive funding).
2. **Drift.** Weights drift with relative performance: ``w * (1 + r_t) / (1 + R_p)``.
   Turnover is measured against these drifted weights, not yesterday's targets, because
   that is what actually has to be traded.
3. **Risk overlay.** The drawdown controller sees equity up to now and returns a multiplier.
4. **Trade** to ``target[t - execution_delay] * multiplier``; pay fees, spread and impact.

With ``execution_delay = 1`` (default) a signal computed from bar ``t`` is traded at the
close of ``t+1``: a conservative allowance for computation and order routing latency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.backtest.costs import CostModel, LinearCostModel, build_cost_model
from alphaforge.backtest.results import BacktestResult
from alphaforge.config.schema import BacktestConfig, CostConfig, RiskConfig
from alphaforge.data.market_data import MarketData
from alphaforge.portfolio.risk import DrawdownController
from alphaforge.utils.logging import get_logger
from alphaforge.utils.time import to_utc

logger = get_logger(__name__)


def mark_to_market(
    weights: np.ndarray, returns: np.ndarray, funding: np.ndarray
) -> tuple[float, float, float, np.ndarray]:
    """Apply one bar of price moves and funding to a book of NAV weights.

    Shared by the backtest loop and :class:`alphaforge.live.PaperBroker` so that simulated
    and paper accounts cannot drift apart through duplicated accounting.

    Returns:
        ``(price_pnl, funding_pnl, growth, drifted_weights)`` where ``growth`` is the NAV
        multiplier of the bar. A non-positive ``growth`` means the book was wiped out and the
        drifted weights are zero.
    """
    price_pnl = float(weights @ returns)
    funding_pnl = -float(weights @ funding)
    growth = 1.0 + price_pnl + funding_pnl
    if growth <= 0:
        return price_pnl, funding_pnl, growth, np.zeros_like(weights)
    return price_pnl, funding_pnl, growth, weights * (1.0 + returns) / growth


class BacktestEngine:
    """Simulates a target-weight portfolio with realistic frictions.

    Args:
        cost_model: Transaction cost model (default: linear, 5.5 bps).
        execution_delay: Bars between signal and execution.
        initial_capital: Starting NAV (matters for market impact).
        apply_funding: Charge perpetual funding on positions.
        drawdown_controller: Optional path-dependent exposure overlay.
        adv_lookback: Bars of dollar volume averaged into ADV for impact.
        vol_halflife: Half-life of the per-bar volatility used by impact.
        drawdown_basis: ``"shadow"`` (default) feeds the controller the NAV of the same book
            at full exposure, simulated alongside; ``"realized"`` feeds it the overlaid NAV,
            whose drawdown freezes once exposure is cut (see docs/METHODOLOGY.md section 6).
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        execution_delay: int = 1,
        initial_capital: float = 1_000_000.0,
        apply_funding: bool = True,
        drawdown_controller: DrawdownController | None = None,
        adv_lookback: int = 20,
        vol_halflife: int = 20,
        drawdown_basis: str = "shadow",
    ) -> None:
        if execution_delay < 0:
            raise ValueError("execution_delay must be >= 0")
        if drawdown_basis not in ("shadow", "realized"):
            raise ValueError("drawdown_basis must be 'shadow' or 'realized'")
        self.cost_model = cost_model or LinearCostModel()
        self.execution_delay = execution_delay
        self.initial_capital = initial_capital
        self.apply_funding = apply_funding
        self.drawdown_controller = drawdown_controller
        self.adv_lookback = adv_lookback
        self.vol_halflife = vol_halflife
        self.drawdown_basis = drawdown_basis

    @classmethod
    def from_config(
        cls, backtest: BacktestConfig, costs: CostConfig, risk: RiskConfig | None = None
    ) -> BacktestEngine:
        controller = None
        if risk is not None and (
            risk.max_drawdown is not None or risk.drawdown_soft_limit is not None
        ):
            controller = DrawdownController(
                risk.max_drawdown, risk.drawdown_soft_limit, risk.cooldown_bars
            )
        return cls(
            cost_model=build_cost_model(costs),
            execution_delay=backtest.execution_delay,
            initial_capital=backtest.initial_capital,
            apply_funding=backtest.apply_funding,
            drawdown_controller=controller,
            adv_lookback=costs.adv_lookback,
            vol_halflife=costs.vol_halflife,
            drawdown_basis=risk.drawdown_basis if risk is not None else "shadow",
        )

    def trade(
        self, drifted: np.ndarray, nav: float, target: np.ndarray, vol: np.ndarray, adv: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        """Rebalance drifted weights to ``target`` at the close: ``(nav_post, cost, trades)``.

        Shared by the real book, the shadow book and :class:`alphaforge.live.PaperBroker`, so the
        three cannot diverge through duplicated cost arithmetic.
        """
        if nav <= 0:
            return 0.0, 0.0, np.zeros_like(drifted)
        trades = target - drifted
        cost = float(self.cost_model.compute(trades, nav, vol, adv).sum())
        return nav * (1.0 - cost), cost, trades

    def cost_inputs(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Per-bar volatility and average dollar volume fed to the cost model.

        Both are trailing estimators (row ``t`` uses bars <= ``t``), so a live caller that
        only has data up to now gets exactly the values the backtest used at that bar.
        Missing vol falls back to the cross-sectional median, then to 2% per bar.
        """
        vol = data.returns().ewm(halflife=self.vol_halflife, min_periods=5).std()
        vol = vol.ffill()
        # Row-wise broadcast; ``vol.T.fillna(median).T`` gives the same values ~300x slower.
        vol = vol.where(vol.notna(), vol.median(axis=1), axis=0).fillna(0.02)
        adv = data.dollar_volume().rolling(self.adv_lookback, min_periods=1).mean()
        return vol, adv

    def run(
        self,
        target_weights: pd.DataFrame,
        data: MarketData,
        start: str | pd.Timestamp | None = None,
        name: str = "portfolio",
    ) -> BacktestResult:
        """Simulate ``target_weights`` (fraction of NAV per asset, decided at each close).

        Args:
            target_weights: Desired weights; rows are decision times.
            data: Market data (risk inputs are estimated on the full history).
            start: First bar of the evaluation; earlier bars only warm up estimators.
            name: Label of the result.
        """
        index = data.index
        symbols = data.close.columns
        targets = target_weights.reindex(index=index, columns=symbols).fillna(0.0)
        desired = targets.shift(self.execution_delay).fillna(0.0).to_numpy()
        tradable = data.close.notna().to_numpy()
        desired = np.where(tradable, desired, 0.0)

        R = data.returns().fillna(0.0).to_numpy()
        F = (
            data.funding.reindex(index=index, columns=symbols).fillna(0.0).to_numpy()
            if self.apply_funding and data.funding is not None
            else np.zeros_like(R)
        )
        vol_frame, adv_frame = self.cost_inputs(data)
        vol, adv = vol_frame.to_numpy(), adv_frame.to_numpy()

        first = 0 if start is None else int(index.searchsorted(to_utc(start)))
        n_bars, n_assets = len(index) - first, len(symbols)
        if n_bars <= 0:
            raise ValueError("start is after the last bar")

        net = np.zeros(n_bars)
        gross = np.zeros(n_bars)
        funding = np.zeros(n_bars)
        costs = np.zeros(n_bars)
        turnover = np.zeros(n_bars)
        risk_scale = np.ones(n_bars)
        positions = np.zeros((n_bars, n_assets))
        equity = np.zeros(n_bars)

        controller = self.drawdown_controller
        if controller is not None:
            controller.reset()
        use_shadow = controller is not None and self.drawdown_basis == "shadow"
        shadow_equity = np.zeros(n_bars)

        nav = self.initial_capital
        w = np.zeros(n_assets)
        nav_s, w_s = self.initial_capital, np.zeros(n_assets)  # shadow book: full exposure
        for i, t in enumerate(range(first, len(index))):
            price_pnl, funding_pnl, growth, w_drift = mark_to_market(w, R[t], F[t])
            if growth <= 0:
                logger.error("%s: NAV wiped out at %s", name, index[t])
                nav_pre = 0.0
            else:
                nav_pre = nav * growth

            basis = nav_pre
            if use_shadow:
                _, _, growth_s, drift_s = mark_to_market(w_s, R[t], F[t])
                basis = nav_s * growth_s if growth_s > 0 else 0.0
                full = desired[t] if basis > 0 else np.zeros(n_assets)
                nav_s, _, _ = self.trade(drift_s, basis, full, vol[t], adv[t])
                w_s = full
                shadow_equity[i] = nav_s

            if controller is None or nav_pre <= 0:
                scale = 1.0
            else:
                scale = controller.update(basis) if basis > 0 else 0.0
            target = desired[t] * scale if nav_pre > 0 else np.zeros(n_assets)
            nav_post, cost, trades = self.trade(w_drift, nav_pre, target, vol[t], adv[t])

            net[i] = nav_post / nav - 1.0 if nav > 0 else 0.0
            gross[i] = price_pnl
            funding[i] = funding_pnl
            costs[i] = cost * growth
            turnover[i] = float(np.abs(trades).sum())
            risk_scale[i] = scale
            positions[i] = target
            equity[i] = nav_post
            nav = nav_post
            w = target

        idx = index[first:]
        pos = pd.DataFrame(positions, index=idx, columns=symbols)
        return BacktestResult(
            name=name,
            returns=pd.Series(net, index=idx, name="returns"),
            gross_returns=pd.Series(gross, index=idx, name="gross_returns"),
            funding=pd.Series(funding, index=idx, name="funding"),
            costs=pd.Series(costs, index=idx, name="costs"),
            turnover=pd.Series(turnover, index=idx, name="turnover"),
            positions=pos,
            equity=pd.Series(equity, index=idx, name="equity"),
            gross_leverage=pos.abs().sum(axis=1).rename("gross_leverage"),
            net_leverage=pos.sum(axis=1).rename("net_leverage"),
            risk_scale=pd.Series(risk_scale, index=idx, name="risk_scale"),
            periods_per_year=data.periods_per_year,
            shadow_equity=pd.Series(shadow_equity, index=idx, name="shadow_equity")
            if use_shadow
            else None,
        )
