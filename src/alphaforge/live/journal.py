"""Structured, append-only event journal (JSON lines) for live and testnet trading.

Every decision a live broker takes (reconciliation, order sent, order found already placed,
fill, kill switch) is one JSON object per line, with a UTC timestamp and an event name. It is
the audit trail: grep-able, machine-readable, and it survives restarts, which plain log text
does not do well. Each event is also mirrored to the standard logger.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alphaforge.utils.logging import get_logger

logger = get_logger("alphaforge.live.journal")


class EventJournal:
    """Append-only JSON-lines journal. ``path=None`` keeps events in memory only (tests)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = None if path is None else Path(path)
        self.events: list[dict[str, Any]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, level: str = "info", **fields: Any) -> dict[str, Any]:
        """Record ``event`` with ``fields``; returns the stored record."""
        record = {"ts": datetime.now(UTC).isoformat(), "event": event, "level": level, **fields}
        line = json.dumps(record, default=str, sort_keys=False)
        self.events.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        logger.log(logging.getLevelName(level.upper()), line)
        return record

    def read(self) -> Iterator[dict[str, Any]]:
        """All events, from the file when there is one (survives restarts)."""
        if self.path is None or not self.path.exists():
            yield from self.events
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
