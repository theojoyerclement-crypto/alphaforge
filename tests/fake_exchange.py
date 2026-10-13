# ruff: noqa: N818  (exception names mirror CCXT's on purpose)
"""A small in-memory stand-in for a CCXT futures exchange in sandbox mode (no network).

It keeps a USDT margin account with one-way positions, fills market orders at the mark price
with a taker fee, enforces amount precision, minimum notional and unique client order ids, and
can inject failures: network errors, a timeout *after* the order was accepted, rejections.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any


def fake_errors() -> SimpleNamespace:
    """Exception hierarchy mirroring CCXT's (no ccxt install needed)."""

    class BaseError(Exception):
        pass

    class NetworkError(BaseError):
        pass

    class RequestTimeout(NetworkError):
        pass

    class ExchangeError(BaseError):
        pass

    class InvalidOrder(ExchangeError):
        pass

    class DuplicateOrderId(InvalidOrder):
        pass

    class OrderNotFound(InvalidOrder):
        pass

    class InsufficientFunds(ExchangeError):
        pass

    classes = (
        BaseError,
        NetworkError,
        RequestTimeout,
        ExchangeError,
        InvalidOrder,
        DuplicateOrderId,
        OrderNotFound,
        InsufficientFunds,
    )
    return SimpleNamespace(**{c.__name__: c for c in classes})


ERRORS = fake_errors()


class FakeExchange:
    """Sandboxed futures account: ``marks`` are the prices orders fill at."""

    def __init__(
        self,
        marks: dict[str, float],
        cash: float = 10_000.0,
        step: float = 0.001,
        min_cost: float = 5.0,
    ) -> None:
        self.isSandboxModeEnabled = True
        self.marks = dict(marks)
        self.cash = cash  # wallet balance, realised PnL and fees included
        self.step, self.min_cost, self.fee_rate = step, min_cost, 0.0004
        self.positions: dict[str, tuple[float, float]] = {}  # symbol -> (amount, entry price)
        self.orders: dict[str, dict[str, Any]] = {}  # client order id -> order
        self.create_calls = 0
        self.fail_create: list[Exception] = []  # raised before the order exists
        self.timeout_after_accept = 0  # accept the order, then raise RequestTimeout
        self.fail_fetch: list[Exception] = []
        self.lagged_positions: dict[str, tuple[float, float]] | None = None  # replication lag

    # -- market metadata ---------------------------------------------------------------------

    def market(self, symbol: str) -> dict[str, Any]:
        return {"symbol": symbol, "limits": {"cost": {"min": self.min_cost}}}

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        return f"{math.floor(amount / self.step + 1e-9) * self.step:.6f}"

    # -- account -----------------------------------------------------------------------------

    def unrealised(self) -> float:
        return sum(a * (self.marks[s] - e) for s, (a, e) in self.positions.items())

    def fetch_balance(self) -> dict[str, Any]:
        if self.fail_fetch:
            raise self.fail_fetch.pop(0)
        nav = self.cash + self.unrealised()
        return {"total": {"USDT": self.cash}, "info": {"totalMarginBalance": str(nav)}}

    def fetch_positions(self, symbols: list[str]) -> list[dict[str, Any]]:
        out = []
        view = self.positions if self.lagged_positions is None else self.lagged_positions
        for s, (a, _) in view.items():
            if s in symbols and a != 0:
                out.append({"symbol": s, "contracts": abs(a), "contractSize": 1.0,
                            "side": "long" if a > 0 else "short"})  # fmt: skip
        return out

    # -- orders ------------------------------------------------------------------------------

    def _fill(self, symbol: str, side: str, amount: float, cid: str) -> dict[str, Any]:
        price = self.marks[symbol]
        signed = amount if side == "buy" else -amount
        held, entry = self.positions.get(symbol, (0.0, price))
        new = held + signed
        if held == 0 or math.copysign(1, held) == math.copysign(1, signed):  # open or add
            self.positions[symbol] = (new, (held * entry + signed * price) / new)
        else:  # reduce, close or flip: realise PnL on the closed part
            closed = min(abs(signed), abs(held))
            self.cash += closed * (price - entry) * math.copysign(1, held)
            if abs(new) < 1e-12:
                self.positions.pop(symbol, None)
            elif math.copysign(1, new) == math.copysign(1, held):
                self.positions[symbol] = (new, entry)
            else:
                self.positions[symbol] = (new, price)
        fee = amount * price * self.fee_rate
        self.cash -= fee
        order = {"id": str(len(self.orders) + 1), "clientOrderId": cid, "symbol": symbol,
                 "side": side, "amount": amount, "filled": amount, "average": price,
                 "status": "closed", "fee": {"cost": fee, "currency": "USDT"}}  # fmt: skip
        self.orders[cid] = order
        return order

    def create_order(
        self, symbol: str, type: str, side: str, amount: float, price: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        self.create_calls += 1
        cid = params["clientOrderId"]
        if self.fail_create:
            raise self.fail_create.pop(0)
        if cid in self.orders:
            raise ERRORS.DuplicateOrderId(f"duplicate clientOrderId {cid}")
        if amount * self.marks[symbol] < self.min_cost:
            raise ERRORS.InvalidOrder("notional below minimum")
        order = self._fill(symbol, side, amount, cid)
        if self.timeout_after_accept:
            self.timeout_after_accept -= 1
            raise ERRORS.RequestTimeout("read timed out")  # the venue did accept the order
        return order

    def fetch_order(self, id: Any, symbol: str, params: dict[str, Any]) -> dict[str, Any]:
        cid = params["clientOrderId"]
        if cid not in self.orders:
            raise ERRORS.OrderNotFound(cid)
        return self.orders[cid]
