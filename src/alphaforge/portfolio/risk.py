"""Risk limits and the path-dependent drawdown controller."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PositionLimits:
    """Hard book constraints applied after sizing."""

    max_position: float = 0.30
    max_gross: float = 2.0
    max_net: float = 1.0


def apply_position_limits(weights: pd.DataFrame, limits: PositionLimits) -> pd.DataFrame:
    """Clip single names, then scale rows breaching gross or net exposure limits.

    Clipping comes first, so the gross/net scaling never pushes a name back over its cap.
    """
    w = weights.clip(-limits.max_position, limits.max_position)
    gross = w.abs().sum(axis=1)
    w = w.mul(np.minimum(1.0, limits.max_gross / gross.replace(0.0, np.nan)).fillna(1.0), axis=0)
    net = w.sum(axis=1).abs()
    w = w.mul(np.minimum(1.0, limits.max_net / net.replace(0.0, np.nan)).fillna(1.0), axis=0)
    return w


class DrawdownController:
    """Exposure overlay driven by the strategy's own drawdown.

    * ``dd < soft_limit``: full exposure.
    * ``soft_limit <= dd < max_drawdown``: exposure falls linearly from 1 to 0.
    * ``dd >= max_drawdown``: flat for ``cooldown_bars``, then the high-water mark resets to
      current equity and trading resumes (otherwise a flat book could never recover).

    The controller only sees equity up to the current close, so it is causal by construction.
    """

    def __init__(
        self,
        max_drawdown: float | None = 0.30,
        soft_limit: float | None = 0.15,
        cooldown_bars: int = 20,
    ) -> None:
        if max_drawdown is not None and soft_limit is not None and soft_limit >= max_drawdown:
            raise ValueError("soft_limit must be below max_drawdown")
        self.max_drawdown = max_drawdown
        self.soft_limit = soft_limit
        self.cooldown_bars = cooldown_bars
        self.reset()

    def reset(self) -> None:
        self._hwm: float | None = None
        self._cooldown = 0
        self.breaches = 0

    @property
    def enabled(self) -> bool:
        return self.max_drawdown is not None or self.soft_limit is not None

    def state_dict(self) -> dict[str, float | int | None]:
        """Path-dependent state, so a paper session can stop and resume between bars."""
        return {"hwm": self._hwm, "cooldown": self._cooldown, "breaches": self.breaches}

    def load_state_dict(self, state: Mapping[str, float | int | None]) -> None:
        """Restore what :meth:`state_dict` returned."""
        hwm = state.get("hwm")
        self._hwm = None if hwm is None else float(hwm)
        self._cooldown = int(state.get("cooldown") or 0)
        self.breaches = int(state.get("breaches") or 0)

    def update(self, equity: float) -> float:
        """Feed the latest equity; return the exposure multiplier in ``[0, 1]``."""
        if self._hwm is None:
            self._hwm = equity
        if self._cooldown > 0:
            self._cooldown -= 1
            if self._cooldown == 0:
                self._hwm = equity
            return 0.0
        self._hwm = max(self._hwm, equity)
        dd = 1.0 - equity / self._hwm
        if self.max_drawdown is not None and dd >= self.max_drawdown:
            self.breaches += 1
            self._cooldown = self.cooldown_bars
            if self._cooldown == 0:
                self._hwm = equity
            return 0.0
        if self.soft_limit is not None and dd > self.soft_limit:
            if self.max_drawdown is None:
                return 0.5
            return float(
                max(0.0, 1.0 - (dd - self.soft_limit) / (self.max_drawdown - self.soft_limit))
            )
        return 1.0
