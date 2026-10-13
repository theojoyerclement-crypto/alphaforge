"""Retry policy and the structured event journal used by live brokers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphaforge.live.journal import EventJournal
from alphaforge.live.retry import call_with_retries


class TransientError(Exception):
    pass


class PermanentError(Exception):
    pass


def test_retries_transient_errors_with_exponential_backoff() -> None:
    calls, sleeps = [], []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise TransientError("timeout")
        return "ok"

    out = call_with_retries(
        flaky, retry_on=(TransientError,), attempts=5, base_delay=0.5, sleep=sleeps.append
    )
    assert out == "ok" and len(calls) == 3 and sleeps == [0.5, 1.0]


def test_gives_up_after_attempts_and_caps_the_delay() -> None:
    sleeps: list[float] = []

    def always() -> None:
        raise TransientError("down")

    with pytest.raises(TransientError):
        call_with_retries(
            always,
            retry_on=(TransientError,),
            attempts=6,
            base_delay=1.0,
            max_delay=4.0,
            sleep=sleeps.append,
        )
    assert sleeps == [1.0, 2.0, 4.0, 4.0, 4.0]


def test_permanent_errors_are_not_retried() -> None:
    calls: list[int] = []

    def rejected() -> None:
        calls.append(1)
        raise PermanentError("order rejected")

    with pytest.raises(PermanentError):
        call_with_retries(rejected, retry_on=(TransientError,), sleep=lambda _: None)
    assert len(calls) == 1


def test_journal_writes_one_json_object_per_line_and_reads_back(tmp_path: Path) -> None:
    journal = EventJournal(tmp_path / "live" / "journal.jsonl")
    journal.log("order_sent", symbol="BTC/USDT:USDT", amount=0.01, client_order_id="af-x")
    journal.log("kill_switch", level="critical", reason="max_loss")
    lines = (tmp_path / "live" / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["order_sent", "kill_switch"]
    assert json.loads(lines[1])["level"] == "critical" and "ts" in json.loads(lines[0])
    reopened = EventJournal(tmp_path / "live" / "journal.jsonl")
    assert [e["event"] for e in reopened.read()] == ["order_sent", "kill_switch"]
