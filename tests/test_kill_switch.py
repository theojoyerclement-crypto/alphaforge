"""Kill switch: each trigger, persistence of the halted state, manual reset."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.live.journal import EventJournal
from alphaforge.live.kill_switch import KillSwitch, KillSwitchConfig, KillSwitchTriggeredError

DAY = pd.Timedelta("1D")
NOW = pd.Timestamp("2026-10-13 08:00", tz="UTC")


def test_max_loss_is_measured_from_the_session_high() -> None:
    ks = KillSwitch(KillSwitchConfig(max_loss=0.10))
    for nav in (100.0, 110.0, 105.0, 99.5):  # 9.5% below 110: still fine
        ks.check_loss(nav)
    with pytest.raises(KillSwitchTriggeredError) as info:
        ks.check_loss(98.9)  # 10.1% below 110
    assert info.value.reason == "max_loss" and ks.halted


def test_position_desync_beyond_tolerance() -> None:
    ks = KillSwitch(KillSwitchConfig(max_position_gap=0.05))
    expected = pd.Series({"BTC": 0.30, "ETH": -0.20})
    assert ks.check_positions(pd.Series({"BTC": 0.31, "ETH": -0.21}), expected) == pytest.approx(
        0.02
    )
    with pytest.raises(KillSwitchTriggeredError) as info:  # a 10% position nobody asked for
        ks.check_positions(pd.Series({"BTC": 0.30, "ETH": -0.20, "SOL": 0.10}), expected)
    assert info.value.reason == "position_desync"


def test_stale_data_counts_from_the_bar_close() -> None:
    ks = KillSwitch(KillSwitchConfig(max_stale_bars=2))
    ks.check_data(pd.Timestamp("2026-10-12", tz="UTC"), DAY, NOW)  # closed 8h ago: 0.33 bar
    ks.check_data(pd.Timestamp("2026-10-11", tz="UTC"), DAY, NOW)  # closed 1.33 bars ago
    with pytest.raises(KillSwitchTriggeredError) as info:  # closed 2026-10-11 00:00: 2.33 bars
        ks.check_data(pd.Timestamp("2026-10-10", tz="UTC"), DAY, NOW)
    assert info.value.reason == "stale_data"


def test_halt_persists_across_restarts_until_manual_reset() -> None:
    journal = EventJournal()
    ks = KillSwitch(KillSwitchConfig(max_loss=0.05), journal)
    ks.check_loss(100.0)
    with pytest.raises(KillSwitchTriggeredError):
        ks.check_loss(90.0)
    restarted = KillSwitch(KillSwitchConfig(max_loss=0.05), journal, state=ks.state_dict())
    with pytest.raises(KillSwitchTriggeredError, match="reset by hand") as info:
        restarted.assert_active()
    assert info.value.reason == "halted"
    restarted.reset(operator="theo")
    restarted.assert_active()
    restarted.check_loss(90.0)  # high-water mark restarts after a reset
    assert [e["event"] for e in journal.read()] == ["kill_switch", "kill_switch_reset"]
    assert journal.events[0]["level"] == "critical"


@pytest.mark.parametrize(
    "kwargs", [{"max_loss": 0.0}, {"max_loss": 1.0}, {"max_position_gap": 0}, {"max_stale_bars": 0}]
)
def test_rejects_meaningless_thresholds(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        KillSwitchConfig(**kwargs)
