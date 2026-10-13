"""Testnet broker: the paper trader's decisions sent as real orders to an exchange sandbox.

``CcxtTestnetBroker`` implements the same :class:`~alphaforge.live.broker.Broker` interface as
:class:`~alphaforge.live.broker.PaperBroker`, so nothing upstream changes: strategies, portfolio
construction, the drawdown overlay and :class:`~alphaforge.live.trader.PaperTrader` are the
backtest's own code. What this layer adds is what a real venue requires:

* **Testnet only.** It refuses an exchange object whose sandbox mode is off. API keys come
  from environment variables (:meth:`from_env`), never from code or config files.
* **Idempotent orders.** Each order carries a client order id derived from (experiment, bar,
  symbol, side). Before sending, and after any network error, the broker looks the id up on
  the venue: a timed-out request that the exchange did accept is never sent a second time, and
  re-running a bar after a crash does not double the trades.
* **Reconciliation.** The journal stores the positions the broker expects. At every sync the
  venue's positions are compared with them; a gap beyond tolerance trips the kill switch.
* **Retries** with exponential backoff on network errors only (:mod:`alphaforge.live.retry`).
* **Kill switch** on max loss, position desync and stale data (:mod:`alphaforge.live.kill_switch`).
* **Structured logs**: every decision is a JSON line in the event journal.

Orders are market orders sized from target NAV weights at the last close, rounded down to the
venue's amount precision and skipped below its minimum notional.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.data.market_data import MarketData
from alphaforge.live.broker import Broker
from alphaforge.live.journal import EventJournal
from alphaforge.live.kill_switch import KillSwitch, KillSwitchConfig, KillSwitchTriggeredError
from alphaforge.live.retry import call_with_retries
from alphaforge.utils.time import timeframe_to_timedelta, to_utc

KEY_ENV, SECRET_ENV = "ALPHAFORGE_TESTNET_API_KEY", "ALPHAFORGE_TESTNET_API_SECRET"
_FILL_COLUMNS = [
    "bar", "symbol", "side", "amount", "filled", "price", "weight_before", "weight_after",
    "trade_notional", "cost", "client_order_id", "status",
]  # fmt: skip


class SandboxRequiredError(RuntimeError):
    """Raised when asked to trade on anything that is not a sandbox / testnet."""


def client_order_id(experiment: str, bar: pd.Timestamp, symbol: str, side: str) -> str:
    """Deterministic id, unique per (experiment, bar, symbol, side), venue-safe (<= 36 chars).

    The amount is deliberately excluded: re-running a bar must map to the same id even if the
    recomputed size differs by a rounding step.
    """
    key = f"{experiment}|{to_utc(bar).isoformat()}|{symbol}|{side}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:20]
    base = re.sub(r"[^A-Za-z0-9]", "", symbol.split("/")[0])[:8]
    return f"af-{base}-{side[0]}-{digest}"


class CcxtTestnetBroker(Broker):
    """A :class:`Broker` backed by a CCXT exchange in sandbox mode.

    Args:
        exchange: A CCXT exchange instance (or a test double) with sandbox mode enabled.
        experiment: Experiment name, part of every client order id.
        timeframe: Bar timeframe of the strategy, for the stale-data check.
        state_path: JSON file holding expected positions, last bar and kill-switch state.
        journal: Event journal (JSON lines); defaults to ``<state dir>/journal.jsonl``.
        kill_switch: Configured kill switch; restored from ``state_path`` when present.
        errors: Namespace of the exchange's exception classes (``ccxt`` by default).
        quote: Settlement currency of the account.
        sleep, now: Injected for tests.
    """

    def __init__(
        self,
        exchange: Any,
        experiment: str,
        timeframe: str,
        state_path: str | Path | None = None,
        journal: EventJournal | None = None,
        kill_switch_config: KillSwitchConfig | None = None,
        errors: Any = None,
        quote: str = "USDT",
        retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
    ) -> None:
        if not getattr(exchange, "isSandboxModeEnabled", False):
            raise SandboxRequiredError(
                "the exchange is not in sandbox mode: CcxtTestnetBroker only trades on testnets"
            )
        if errors is None:
            import ccxt

            errors = ccxt
        self.exchange, self.errors = exchange, errors
        self.experiment, self.quote = experiment, quote
        self.bar = timeframe_to_timedelta(timeframe)
        self.retries, self.sleep, self.now = retries, sleep, now
        self.state_path = None if state_path is None else Path(state_path)
        if journal is None and self.state_path is not None:
            journal = EventJournal(self.state_path.parent / "journal.jsonl")
        self.journal = journal or EventJournal()

        saved = self._load_state()
        self.expected: dict[str, float] = {
            k: float(v) for k, v in saved.get("expected", {}).items()
        }
        self.has_history = "expected" in saved
        self._last_bar: str | None = saved.get("last_bar")
        self.kill_switch = KillSwitch(kill_switch_config, self.journal, saved.get("kill_switch"))
        self._nav = 0.0
        self._amounts: dict[str, float] = {}
        self._prices: dict[str, float] = {}

    # -- construction --------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        experiment: str,
        timeframe: str,
        state_path: str | Path,
        exchange_id: str = "binanceusdm",
        kill_switch_config: KillSwitchConfig | None = None,
    ) -> CcxtTestnetBroker:
        """Build a sandboxed CCXT exchange from ``ALPHAFORGE_TESTNET_API_KEY/SECRET``."""
        key, secret = os.environ.get(KEY_ENV), os.environ.get(SECRET_ENV)
        if not key or not secret:
            raise SandboxRequiredError(f"set {KEY_ENV} and {SECRET_ENV} (testnet keys only)")
        import ccxt

        exchange = getattr(ccxt, exchange_id)(
            {"apiKey": key, "secret": secret, "enableRateLimit": True}
        )
        exchange.set_sandbox_mode(True)
        return cls(
            exchange, experiment, timeframe, state_path, kill_switch_config=kill_switch_config
        )

    # -- Broker interface ------------------------------------------------------------------

    @property
    def nav(self) -> float:
        return self._nav

    @property
    def weights(self) -> pd.Series:
        if self._nav <= 0:
            return pd.Series(dtype=float)
        return pd.Series(
            {s: a * self._prices.get(s, np.nan) / self._nav for s, a in self._amounts.items()},
            dtype=float,
        ).dropna()

    @property
    def last_bar(self) -> pd.Timestamp | None:
        return None if self._last_bar is None else to_utc(self._last_bar)

    def sync(self, data: MarketData) -> pd.DataFrame:
        """Read the account, run the kill-switch checks and reconcile positions."""
        self.kill_switch.assert_active()
        bar = data.index[-1]
        self._prices = {s: float(p) for s, p in data.close.iloc[-1].dropna().items()}
        try:
            self.kill_switch.check_data(bar, self.bar, self.now())
            self._refresh(list(data.symbols))
            self.kill_switch.check_loss(self._nav)
            self._reconcile()
        except KillSwitchTriggeredError:
            self._save_state()  # a restart must find the switch halted
            raise
        self._last_bar = bar.isoformat()
        self._save_state()
        self.journal.log("sync", bar=bar, nav=self._nav, positions=self._amounts)
        return pd.DataFrame({"nav": [self._nav]}, index=pd.DatetimeIndex([bar], name="bar"))

    def execute(self, target: pd.Series, data: MarketData) -> pd.DataFrame:
        """Send the market orders that move positions to ``target`` NAV weights."""
        self.kill_switch.assert_active()
        bar = data.index[-1]
        if self.last_bar is None or self.last_bar != bar:
            raise ValueError("sync() the account to the latest bar before executing")
        rows = []
        for symbol in data.symbols:
            price = self._prices.get(symbol)
            current = self._amounts.get(symbol, 0.0)
            weight = float(target.get(symbol, 0.0)) if price else 0.0
            desired = weight * self._nav / price if price else 0.0
            delta = desired - current
            if price is None or delta == 0.0:
                continue
            side = "buy" if delta > 0 else "sell"
            amount = float(self.exchange.amount_to_precision(symbol, abs(delta)))
            min_cost = (self.exchange.market(symbol).get("limits", {}).get("cost") or {}).get("min")
            if amount <= 0 or (min_cost and amount * price < float(min_cost)):
                self.journal.log(
                    "order_skipped", symbol=symbol, amount=abs(delta), reason="below minimum"
                )
                continue
            reduce_only = (
                current != 0 and np.sign(delta) != np.sign(current) and abs(delta) <= abs(current)
            )
            cid = client_order_id(self.experiment, bar, symbol, side)
            order = self._place(symbol, side, amount, cid, bool(reduce_only))
            filled = float(order.get("filled") or 0.0) if order else 0.0
            signed = filled if side == "buy" else -filled
            self.expected[symbol] = self.expected.get(symbol, 0.0) + signed
            fee = float((order or {}).get("fee", {}).get("cost") or 0.0)
            fill_price = float((order or {}).get("average") or price)
            rows.append([
                bar, symbol, side, amount, filled, fill_price,
                current * price / self._nav, (current + signed) * price / self._nav,
                signed * fill_price, fee, cid, (order or {}).get("status", "rejected"),
            ])  # fmt: skip
        self._refresh(list(data.symbols))
        self._save_state()
        return pd.DataFrame(rows, columns=_FILL_COLUMNS)

    # -- orders --------------------------------------------------------------------------

    def _find_order(self, client_id: str, symbol: str) -> dict[str, Any] | None:
        try:
            return call_with_retries(
                lambda: self.exchange.fetch_order(None, symbol, {"clientOrderId": client_id}),
                retry_on=(self.errors.NetworkError,),
                attempts=self.retries,
                sleep=self.sleep,
                label=f"fetch_order {client_id}",
            )
        except self.errors.OrderNotFound:
            return None

    def _place(
        self, symbol: str, side: str, amount: float, client_id: str, reduce_only: bool
    ) -> dict[str, Any] | None:
        """Send one market order at most once, whatever the network does."""
        existing = self._find_order(client_id, symbol)
        if existing is not None:
            self.journal.log(
                "order_already_placed",
                symbol=symbol,
                client_order_id=client_id,
                status=existing.get("status"),
            )
            return existing
        params = {"clientOrderId": client_id, "reduceOnly": reduce_only}
        delay = 0.5
        for attempt in range(1, self.retries + 1):
            self.journal.log(
                "order_sent",
                symbol=symbol,
                side=side,
                amount=amount,
                client_order_id=client_id,
                reduce_only=reduce_only,
                attempt=attempt,
            )
            try:
                order: dict[str, Any] = self.exchange.create_order(
                    symbol, "market", side, amount, None, params
                )
                self.journal.log(
                    "order_filled",
                    symbol=symbol,
                    client_order_id=client_id,
                    filled=order.get("filled"),
                    average=order.get("average"),
                )
                return order
            except self.errors.DuplicateOrderId:
                found = self._find_order(client_id, symbol)
                self.journal.log("order_already_placed", symbol=symbol, client_order_id=client_id)
                return found
            except self.errors.NetworkError as exc:
                # The request may have reached the venue: look before sending again.
                found = self._find_order(client_id, symbol)
                if found is not None:
                    self.journal.log(
                        "order_found_after_error",
                        level="warning",
                        symbol=symbol,
                        client_order_id=client_id,
                        error=str(exc),
                    )
                    return found
                if attempt == self.retries:
                    self.journal.log(
                        "order_failed",
                        level="error",
                        symbol=symbol,
                        client_order_id=client_id,
                        error=str(exc),
                    )
                    raise
                self.sleep(delay)
                delay = min(delay * 2, 8.0)
            except (self.errors.InvalidOrder, self.errors.InsufficientFunds) as exc:
                self.journal.log(
                    "order_rejected",
                    level="error",
                    symbol=symbol,
                    client_order_id=client_id,
                    error=str(exc),
                )
                return None
        return None

    # -- account -------------------------------------------------------------------------

    def _refresh(self, symbols: list[str]) -> None:
        balance = call_with_retries(
            self.exchange.fetch_balance, retry_on=(self.errors.NetworkError,),
            attempts=self.retries, sleep=self.sleep, label="fetch_balance",
        )  # fmt: skip
        # Futures: margin balance (wallet + unrealised PnL) is the account's NAV.
        info = balance.get("info") or {}
        self._nav = float(info.get("totalMarginBalance") or balance["total"][self.quote])
        positions = call_with_retries(
            lambda: self.exchange.fetch_positions(symbols), retry_on=(self.errors.NetworkError,),
            attempts=self.retries, sleep=self.sleep, label="fetch_positions",
        )  # fmt: skip
        amounts: dict[str, float] = {}
        for pos in positions:
            contracts = float(pos.get("contracts") or 0.0) * float(pos.get("contractSize") or 1.0)
            if contracts:
                amounts[pos["symbol"]] = contracts if pos.get("side") == "long" else -contracts
        self._amounts = amounts

    def _reconcile(self) -> None:
        """Compare venue positions with what the journal expects (in NAV weight)."""
        if not self.has_history:
            # First run on this state file: adopt the venue's positions as the baseline.
            self.expected = dict(self._amounts)
            self.has_history = True
            self.journal.log(
                "reconcile_adopted",
                level="warning" if self._amounts else "info",
                positions=self._amounts,
            )
            return
        to_weight = lambda amounts: pd.Series(  # noqa: E731
            {s: a * self._prices.get(s, 0.0) / self._nav for s, a in amounts.items()}, dtype=float
        )
        gap = self.kill_switch.check_positions(to_weight(self._amounts), to_weight(self.expected))
        self.journal.log("reconciled", gap=gap)

    # -- persistence ---------------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        if self.state_path is None or not self.state_path.exists():
            return {}
        raw: dict[str, Any] = json.loads(self.state_path.read_text(encoding="utf-8"))
        if raw.get("experiment") not in (None, self.experiment):
            raise ValueError(f"{self.state_path} belongs to experiment {raw.get('experiment')!r}")
        return raw

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": self.experiment,
            "last_bar": self._last_bar,
            "expected": self.expected,
            "kill_switch": self.kill_switch.state_dict(),
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
