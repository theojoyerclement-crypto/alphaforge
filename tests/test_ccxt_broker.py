"""CcxtTestnetBroker against an in-memory exchange: no network, no ccxt needed."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaforge.data.market_data import MarketData
from alphaforge.live.ccxt_broker import CcxtTestnetBroker, SandboxRequiredError, client_order_id
from alphaforge.live.kill_switch import KillSwitchConfig, KillSwitchTriggeredError
from tests.fake_exchange import ERRORS, FakeExchange

SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]


def _market(prices: dict[str, float], end: str = "2026-10-12", n: int = 5) -> MarketData:
    idx = pd.date_range(end=end, periods=n, freq="1D", tz="UTC", name="timestamp")
    frames = {}
    for s, p in prices.items():
        close = np.full(n, p)
        frames[s] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1e6}, index=idx
        )
    return MarketData.from_frames(frames, "1d")


def _broker(exchange: FakeExchange, tmp_path: Path, **kw) -> CcxtTestnetBroker:  # type: ignore[no-untyped-def]
    return CcxtTestnetBroker(
        exchange,
        "exp",
        "1d",
        state_path=tmp_path / "state.json",
        errors=ERRORS,
        sleep=lambda _: None,
        now=lambda: pd.Timestamp("2026-10-13 08:00", tz="UTC"),
        **kw,
    )


PRICES = {"BTC/USDT:USDT": 50_000.0, "ETH/USDT:USDT": 2_500.0}
TARGET = pd.Series({"BTC/USDT:USDT": 0.5, "ETH/USDT:USDT": -0.25})


def test_refuses_anything_but_a_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live = FakeExchange(PRICES)
    live.isSandboxModeEnabled = False
    with pytest.raises(SandboxRequiredError, match="sandbox"):
        _broker(live, tmp_path)
    monkeypatch.delenv("ALPHAFORGE_TESTNET_API_KEY", raising=False)
    with pytest.raises(SandboxRequiredError, match="ALPHAFORGE_TESTNET_API_KEY"):
        CcxtTestnetBroker.from_env("exp", "1d", tmp_path / "s.json")


def test_trades_to_target_weights_and_journals(tmp_path: Path) -> None:
    ex, data = FakeExchange(PRICES), _market(PRICES)
    broker = _broker(ex, tmp_path)
    broker.sync(data)
    fills = broker.execute(TARGET, data)
    assert list(fills["side"]) == ["buy", "sell"] and (fills["status"] == "closed").all()
    np.testing.assert_allclose(broker.weights.reindex(TARGET.index), TARGET, atol=0.01)
    bar = data.index[-1]
    assert fills.loc[0, "client_order_id"] == client_order_id("exp", bar, SYMBOLS[0], "buy")
    assert len(fills.loc[0, "client_order_id"]) <= 36
    events = [e["event"] for e in broker.journal.read()]
    assert events[:2] == ["reconcile_adopted", "sync"] and events.count("order_filled") == 2


def test_rerunning_a_bar_after_a_crash_sends_nothing_twice(tmp_path: Path) -> None:
    """Crash after the orders were sent, before the state was saved, and the positions endpoint
    lags (still shows the account flat): the re-run recomputes the same orders, finds them by
    client order id and sends nothing."""
    ex, data = FakeExchange(PRICES), _market(PRICES)
    first = _broker(ex, tmp_path)
    first.sync(data)
    shutil.copy(tmp_path / "state.json", tmp_path / "before_execute.json")
    first.execute(TARGET, data)
    sent = ex.create_calls
    shutil.copy(tmp_path / "before_execute.json", tmp_path / "state.json")  # the crash
    ex.lagged_positions = {}
    second = _broker(ex, tmp_path)
    second.sync(data)
    second.execute(TARGET, data)
    assert ex.create_calls == sent, "orders found by client id must not be re-sent"
    assert [e["event"] for e in second.journal.read()].count("order_already_placed") == 2
    ex.lagged_positions = None
    second.sync(data)  # expectations now match the venue: no desync


def test_timeout_after_the_venue_accepted_is_not_resent(tmp_path: Path) -> None:
    ex, data = FakeExchange(PRICES), _market(PRICES)
    ex.timeout_after_accept = 1
    broker = _broker(ex, tmp_path)
    broker.sync(data)
    broker.execute(TARGET, data)
    assert ex.create_calls == 2  # one per symbol: the timed-out BTC order was looked up, not resent
    assert ex.positions["BTC/USDT:USDT"][0] == pytest.approx(0.1)
    assert "order_found_after_error" in [e["event"] for e in broker.journal.read()]


def test_network_error_before_acceptance_is_retried_once(tmp_path: Path) -> None:
    ex, data = FakeExchange(PRICES), _market(PRICES)
    ex.fail_create = [ERRORS.NetworkError("connection reset")]
    broker = _broker(ex, tmp_path)
    broker.sync(data)
    broker.execute(TARGET, data)
    assert ex.positions["BTC/USDT:USDT"][0] == pytest.approx(0.1)  # filled exactly once
    assert ex.create_calls == 3


def test_rejected_order_is_journaled_and_does_not_change_expectations(tmp_path: Path) -> None:
    ex, data = FakeExchange(PRICES), _market(PRICES)
    ex.fail_create = [ERRORS.InsufficientFunds("margin is insufficient")]
    broker = _broker(ex, tmp_path)
    broker.sync(data)
    fills = broker.execute(TARGET, data)
    assert fills.loc[0, "status"] == "rejected" and "BTC/USDT:USDT" not in ex.positions
    broker.sync(_market(PRICES, end="2026-10-12"))  # expectations match the venue: no desync


def test_foreign_trade_trips_position_desync_and_halt_persists(tmp_path: Path) -> None:
    ex, data = FakeExchange(PRICES), _market(PRICES)
    broker = _broker(ex, tmp_path)
    broker.sync(data)
    broker.execute(TARGET, data)
    ex.positions["ETH/USDT:USDT"] = (2.0, 2_500.0)  # someone flipped ETH by hand
    with pytest.raises(KillSwitchTriggeredError) as info:
        broker.sync(data)
    assert info.value.reason == "position_desync"
    restarted = _broker(ex, tmp_path)
    with pytest.raises(KillSwitchTriggeredError, match="reset by hand"):
        restarted.sync(data)


def test_stale_data_and_max_loss_stop_trading(tmp_path: Path) -> None:
    ex = FakeExchange(PRICES)
    stale = _broker(ex, tmp_path / "a")
    with pytest.raises(KillSwitchTriggeredError, match="stale_data"):
        stale.sync(_market(PRICES, end="2026-10-01"))

    data = _market(PRICES)
    broker = _broker(ex, tmp_path / "b", kill_switch_config=KillSwitchConfig(max_loss=0.05))
    broker.sync(data)
    broker.execute(pd.Series({"BTC/USDT:USDT": 1.0}), data)
    ex.marks["BTC/USDT:USDT"] = 45_000.0  # -10% on a fully invested book
    with pytest.raises(KillSwitchTriggeredError, match="max_loss"):
        broker.sync(_market({**PRICES, "BTC/USDT:USDT": 45_000.0}))
