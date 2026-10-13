"""Exchange data via CCXT (Binance, Bybit, OKX, ...): paginated OHLCV and funding history."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from alphaforge.data.schema import OHLCV_COLUMNS
from alphaforge.data.sources.base import DataSource
from alphaforge.utils.logging import get_logger
from alphaforge.utils.time import timeframe_to_timedelta

logger = get_logger(__name__)

_DEFAULT_START = "2020-01-01"


def _to_ms(ts: str | pd.Timestamp | None, default: str | None = None) -> int:
    value = ts if ts is not None else default
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return int(stamp.timestamp() * 1000)


class CCXTSource(DataSource):
    """Paginated historical downloads through a CCXT exchange client.

    Args:
        exchange_id: CCXT exchange id (``"binance"``, ``"bybit"``, ``"okx"``...).
        market_type: ``"spot"`` or ``"swap"`` (perpetuals; required for funding).
        max_retries: Retries on transient network errors (exponential back-off).
        page_limit: Bars per request.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        market_type: str = "swap",
        max_retries: int = 5,
        page_limit: int = 1000,
        timeout_ms: int = 30_000,
    ) -> None:
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("CCXTSource requires `pip install ccxt`") from exc
        exchange_cls = getattr(ccxt, exchange_id, None)
        if exchange_cls is None:
            raise ValueError(f"Unknown CCXT exchange {exchange_id!r}")
        self._ccxt = ccxt
        self.exchange = exchange_cls(
            {
                "enableRateLimit": True,
                "timeout": timeout_ms,
                "options": {"defaultType": market_type},
            }
        )
        self.name = f"ccxt_{exchange_id}_{market_type}"
        self.max_retries = max_retries
        self.page_limit = page_limit

    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except (self._ccxt.NetworkError, self._ccxt.ExchangeNotAvailable) as exc:
                if attempt == self.max_retries:
                    raise
                logger.warning(
                    "%s failed (%s); retry %d in %.0fs", fn.__name__, exc, attempt, delay
                )
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        since = _to_ms(start, _DEFAULT_START)
        end_ms = _to_ms(end)
        step = int(timeframe_to_timedelta(timeframe).total_seconds() * 1000)
        rows: list[list[float]] = []
        while since <= end_ms:
            batch = self._call(
                self.exchange.fetch_ohlcv, symbol, timeframe, since=since, limit=self.page_limit
            )
            if not batch:
                break
            rows.extend(batch)
            last = int(batch[-1][0])
            if last < since:
                break
            since = last + step
        frame = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
        frame.index = pd.to_datetime(frame.pop("timestamp"), unit="ms", utc=True)
        frame.index.name = "timestamp"
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frame = frame.loc[: pd.Timestamp(end_ms, unit="ms", tz="UTC")]
        # The exchange returns the still-forming bar; its "close" is a mid-bar price that would
        # be cached as if final. Keep only bars whose interval has fully elapsed.
        complete = frame.index + pd.Timedelta(milliseconds=step) <= pd.Timestamp.now(tz="UTC")
        if not complete.all():
            logger.info(
                "Dropping %d still-forming %s bar(s) for %s", (~complete).sum(), timeframe, symbol
            )
        frame = frame[complete]
        logger.info("Fetched %d %s bars for %s", len(frame), timeframe, symbol)
        return frame.astype(float)

    def fetch_funding(
        self,
        symbol: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.Series | None:
        if not self.exchange.has.get("fetchFundingRateHistory"):
            logger.warning("%s does not expose funding history", self.name)
            return None
        since = _to_ms(start, _DEFAULT_START)
        end_ms = _to_ms(end)
        records: list[dict[str, Any]] = []
        while since <= end_ms:
            batch = self._call(
                self.exchange.fetch_funding_rate_history, symbol, since=since, limit=self.page_limit
            )
            if not batch:
                break
            records.extend(batch)
            last = int(batch[-1]["timestamp"])
            if last < since:
                break
            since = last + 1
        if not records:
            return None
        series = pd.Series(
            [float(r["fundingRate"]) for r in records],
            index=pd.to_datetime([int(r["timestamp"]) for r in records], unit="ms", utc=True),
            name="funding_rate",
        )
        series = series[~series.index.duplicated(keep="last")].sort_index()
        return series.loc[: pd.Timestamp(end_ms, unit="ms", tz="UTC")]
