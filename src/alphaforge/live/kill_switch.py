"""Kill switch: stop trading when the account, the positions or the data cannot be trusted.

Three independent triggers, each checked before any order is sent:

* **max loss**: NAV has fallen more than ``max_loss`` below its high-water mark for the session;
* **position desync**: the venue's positions differ from what the journal expected by more than
  ``max_position_gap`` of NAV in total (a fill we did not see, a manual trade, a lost order);
* **stale data**: the latest closed bar is older than ``max_stale_bars`` bars, so signals would
  be computed on an old market.

Once tripped, the switch stays **halted**, and the state is persisted, until a human calls
:meth:`KillSwitch.reset`. Restarting the process must not silently resume trading after the
condition that stopped it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from alphaforge.live.journal import EventJournal


@dataclass(frozen=True)
class KillSwitchConfig:
    """Thresholds; defaults are deliberately conservative for a testnet run."""

    max_loss: float = 0.10  # fraction of NAV below the session high-water mark
    max_position_gap: float = 0.05  # sum over symbols of |actual - expected| NAV weight
    max_stale_bars: int = 2

    def __post_init__(self) -> None:
        if not 0 < self.max_loss < 1:
            raise ValueError("max_loss must be in (0, 1)")
        if self.max_position_gap <= 0 or self.max_stale_bars < 1:
            raise ValueError("max_position_gap must be > 0 and max_stale_bars >= 1")


class KillSwitchTriggeredError(RuntimeError):
    """Trading is halted; ``reason`` is one of ``max_loss``, ``position_desync``, ``stale_data``
    or ``halted`` (tripped earlier and not reset)."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"kill switch: {reason}: {detail}")
        self.reason = reason


class KillSwitch:
    """Checks the three conditions and holds the halted state."""

    def __init__(
        self,
        config: KillSwitchConfig | None = None,
        journal: EventJournal | None = None,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or KillSwitchConfig()
        self.journal = journal or EventJournal()
        state = state or {}
        self.halted: bool = bool(state.get("halted", False))
        self.reason: str | None = state.get("reason")
        self.high_water: float | None = state.get("high_water")

    # -- state ---------------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {"halted": self.halted, "reason": self.reason, "high_water": self.high_water}

    def reset(self, operator: str) -> None:
        """Manual reset after the cause was investigated. The high-water mark restarts too."""
        self.journal.log(
            "kill_switch_reset", level="warning", operator=operator, reason=self.reason
        )
        self.halted, self.reason, self.high_water = False, None, None

    def _trip(self, reason: str, detail: str, **fields: Any) -> None:
        self.halted, self.reason = True, reason
        self.journal.log("kill_switch", level="critical", reason=reason, detail=detail, **fields)
        raise KillSwitchTriggeredError(reason, detail)

    # -- checks --------------------------------------------------------------------------

    def assert_active(self) -> None:
        if self.halted:
            raise KillSwitchTriggeredError(
                "halted", f"tripped earlier ({self.reason}); investigate, then reset by hand"
            )

    def check_data(self, last_bar: pd.Timestamp, bar: pd.Timedelta, now: pd.Timestamp) -> None:
        """The newest *closed* bar must be recent: a bar opening at t closes at t + bar."""
        age_in_bars = (now - (last_bar + bar)) / bar
        if age_in_bars > self.config.max_stale_bars:
            self._trip(
                "stale_data",
                f"last closed bar {last_bar} is {age_in_bars:.1f} bars old",
                last_bar=last_bar,
                age_in_bars=round(age_in_bars, 2),
            )

    def check_loss(self, nav: float) -> None:
        self.high_water = nav if self.high_water is None else max(self.high_water, nav)
        loss = 1.0 - nav / self.high_water if self.high_water > 0 else 1.0
        if loss >= self.config.max_loss:
            self._trip(
                "max_loss",
                f"NAV {nav:,.2f} is {loss:.1%} below the session high {self.high_water:,.2f}",
                nav=nav,
                high_water=self.high_water,
            )

    def check_positions(self, actual: pd.Series, expected: pd.Series) -> float:
        """Total absolute weight gap between venue and journal; trips above the tolerance."""
        symbols = actual.index.union(expected.index)
        gap = float(
            (actual.reindex(symbols).fillna(0.0) - expected.reindex(symbols).fillna(0.0))
            .abs()
            .sum()
        )
        if gap > self.config.max_position_gap:
            detail = {s: [float(actual.get(s, 0.0)), float(expected.get(s, 0.0))] for s in symbols}
            self._trip("position_desync", f"weight gap {gap:.2%} of NAV", gap=gap, positions=detail)
        return gap
