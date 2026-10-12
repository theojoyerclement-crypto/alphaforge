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
    ) -> None:
        if execution_delay < 0:
            raise ValueError("execution_delay must be >= 0")
        self.cost_model = cost_model or LinearCostModel()
        self.execution_delay = execution_delay
        self.initial_capital = initial_capital
        self.apply_funding = apply_funding
        self.drawdown_controller = drawdown_controller
        self.adv_lookback = adv_lookback
        self.vol_halflife = vol_halflife

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
        )

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
        vol = data.returns().ewm(halflife=self.vol_halflife, min_periods=5).std()
        vol = vol.ffill()
        vol = vol.T.fillna(vol.median(axis=1)).T.fillna(0.02).to_numpy()
        adv = data.dollar_volume().rolling(self.adv_lookback, min_periods=1).mean().to_numpy()

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

        nav = self.initial_capital
        w = np.zeros(n_assets)
        for i, t in enumerate(range(first, len(index))):
            price_pnl = float(w @ R[t])
            funding_pnl = -float(w @ F[t])
            bar_ret = price_pnl + funding_pnl
            growth = 1.0 + bar_ret
            if growth <= 0:
                logger.error("%s: NAV wiped out at %s", name, index[t])
                w_drift = np.zeros(n_assets)
                nav_pre = 0.0
            else:
                w_drift = w * (1.0 + R[t]) / growth
                nav_pre = nav * growth

            scale = controller.update(nav_pre) if controller is not None and nav_pre > 0 else 1.0
            target = desired[t] * scale if nav_pre > 0 else np.zeros(n_assets)
            trades = target - w_drift
            cost = (
                float(self.cost_model.compute(trades, nav_pre, vol[t], adv[t]).sum())
                if nav_pre > 0
                else 0.0
            )
            nav_post = nav_pre * (1.0 - cost)

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
        )
