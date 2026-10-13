"""Local Parquet store: ``<root>/<source>/<symbol>/<timeframe>.<kind>.parquet``.

Writes merge with existing files (new rows win on overlap) and update a JSON sidecar with
row count, date range and a content hash, so every dataset used by a run is traceable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.utils.reproducibility import hash_dataframe

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_symbol(symbol: str) -> str:
    """``"BTC/USDT:USDT"`` -> ``"BTC-USDT_USDT"`` (filesystem-safe and reversible enough)."""
    return _SAFE.sub("_", symbol.replace("/", "-"))


class ParquetStore:
    """Parquet-backed time-series store."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, source: str, symbol: str, timeframe: str, kind: str = "ohlcv") -> Path:
        return self.root / source / sanitize_symbol(symbol) / f"{timeframe}.{kind}.parquet"

    def exists(self, source: str, symbol: str, timeframe: str, kind: str = "ohlcv") -> bool:
        return self.path(source, symbol, timeframe, kind).exists()

    def write(
        self,
        frame: pd.DataFrame,
        source: str,
        symbol: str,
        timeframe: str,
        kind: str = "ohlcv",
        merge: bool = True,
        meta: Mapping[str, Any] | None = None,
    ) -> Path:
        """Persist ``frame`` (UTC-indexed); merge with any existing data when ``merge``.

        ``meta`` (e.g. exchange, client library version, requested range) is stored in the
        JSON sidecar next to the row count, date range and content hash.
        """
        target = self.path(source, symbol, timeframe, kind)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = frame
        if merge and target.exists():
            existing = pd.read_parquet(target)
            data = pd.concat([existing, frame])
            data = data[~data.index.duplicated(keep="last")]
        data = data.sort_index()
        data.to_parquet(target)
        meta = {
            "symbol": symbol,
            "source": source,
            "timeframe": timeframe,
            "kind": kind,
            "rows": len(data),
            "start": str(data.index.min()) if len(data) else None,
            "end": str(data.index.max()) if len(data) else None,
            "hash": hash_dataframe(data),
            "updated_at": datetime.now(UTC).isoformat(),
            **dict(meta or {}),
        }
        target.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return target

    def metadata(
        self, source: str, symbol: str, timeframe: str, kind: str = "ohlcv"
    ) -> dict[str, Any] | None:
        """The JSON sidecar of a stored file (``None`` if absent)."""
        sidecar = self.path(source, symbol, timeframe, kind).with_suffix(".json")
        if not sidecar.exists():
            return None
        meta: dict[str, Any] = json.loads(sidecar.read_text(encoding="utf-8"))
        return meta

    def read(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        kind: str = "ohlcv",
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Load a stored frame, optionally restricted to ``[start, end]``."""
        target = self.path(source, symbol, timeframe, kind)
        if not target.exists():
            raise FileNotFoundError(f"No stored {kind} for {symbol} {timeframe} in {source}")
        frame = pd.read_parquet(target)
        return frame.loc[start:end]

    def list_symbols(self, source: str) -> list[str]:
        base = self.root / source
        return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []
