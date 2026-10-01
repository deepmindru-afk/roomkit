"""Tests for ToolAuditEntry, JSONLToolAuditor, and audit wrappers."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import pytest

from roomkit.core.exceptions import ToolRefusedError, ToolTimeoutError, UnservedToolCallError
from roomkit.orchestration.tool_audit import (
    JSONLToolAuditor,
    ToolAuditEntry,
    audit_tool_handler,
)
from roomkit.providers.ai.base import AIImagePart, AITextPart
from roomkit.tools.timeout import answer_within

# ---------------------------------------------------------------------------
# ToolAuditEntry model
# ---------------------------------------------------------------------------


def test_audit_entry_model_dump() -> None:
    entry = ToolAuditEntry(
        ts="t",
        agent_id="a",
        tool_name="search",
        arguments={"q": "hello"},
        result="found",
        status="ok",
        duration_ms=42.5,
    )
    d = entry.model_dump()
    assert d["tool_name"] == "search"
    assert d["status"] == "ok"
    assert d["metadata"] == {}


def test_audit_entry_model_validate() -> None:
    data = {
        "ts": "t",
        "agent_id": "a",
        "tool_name": "search",
        "arguments": {},
        "result": "ok",
        "status": "ok",
        "duration_ms": 10,
    }
    entry = ToolAuditEntry.model_validate(data)
    assert entry.tool_name == "search"


# ---------------------------------------------------------------------------
# JSONLToolAuditor
# ---------------------------------------------------------------------------


def test_jsonl_auditor_record_and_summary(tmp_path: Path) -> None:
    p = tmp_path / "audit.jsonl"
    auditor = JSONLToolAuditor(p)
    auditor.record(
        ToolAuditEntry(
            ts="t",
            agent_id="a",
            tool_name="search",
            arguments={"q": "x"},
            result="done",
            status="ok",
            duration_ms=10,
        )
    )
    assert len(auditor.entries) == 1

    lines = p.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["tool_name"] == "search"

    summary = auditor.summary()
    assert "search" in summary
    assert "1 calls" in summary


def test_jsonl_auditor_empty_summary(tmp_path: Path) -> None:
    p = tmp_path / "audit.jsonl"
    auditor = JSONLToolAuditor(p)
    assert "No tool calls recorded" in auditor.summary()


# ---------------------------------------------------------------------------
# audit_tool_handler
# ---------------------------------------------------------------------------


async def test_audit_tool_handler_records(tmp_path: Path) -> None:
    auditor = JSONLToolAuditor(tmp_path / "audit.jsonl")

    async def handler(name: str, args: dict[str, Any]) -> str:
        return '{"status": "ok", "data": "hello"}'

    wrapped = audit_tool_handler(handler, auditor, "test-agent")
    result = await wrapped("my_tool", {"x": 1})
    assert "hello" in result
    assert len(auditor.entries) == 1
    assert auditor.entries[0].status == "ok"
    assert auditor.entries[0].tool_name == "my_tool"


async def test_audit_tool_handler_detects_failed(tmp_path: Path) -> None:
    auditor = JSONLToolAuditor(tmp_path / "audit.jsonl")

    async def handler(name: str, args: dict[str, Any]) -> str:
        return '{"status": "failed", "error": "not found"}'

    wrapped = audit_tool_handler(handler, auditor, "a")
    await wrapped("tool", {})
    assert auditor.entries[0].status == "failed"


async def test_audit_tool_handler_error(tmp_path: Path) -> None:
    auditor = JSONLToolAuditor(tmp_path / "audit.jsonl")

    async def handler(name: str, args: dict[str, Any]) -> str:
        raise ValueError("boom")

    wrapped = audit_tool_handler(handler, auditor, "a")
    with contextlib.suppress(ValueError):
        await wrapped("tool", {})
    assert auditor.entries[0].status == "error"
    assert "boom" in auditor.entries[0].result


async def test_audit_tool_handler_truncates(tmp_path: Path) -> None:
    auditor = JSONLToolAuditor(tmp_path / "audit.jsonl")

    async def handler(name: str, args: dict[str, Any]) -> str:
        return "x" * 1000

    wrapped = audit_tool_handler(handler, auditor, "a")
    result = await wrapped("tool", {})
    # The returned result is NOT truncated
    assert len(result) == 1000
    # But the recorded entry IS truncated
    assert len(auditor.entries[0].result) == 500


# -- The wrapper hands the channel the answer itself (RFC §15.8.1, RMK-305) --


@pytest.mark.parametrize(
    ("answer", "recorded", "status"),
    [
        (
            [AITextPart(text="a chart"), AIImagePart(url="https://x/c.png")],
            "a chart\n[image]",
            "ok",
        ),
        ({"found": 3}, '{"found": 3}', "ok"),
        ({"error": "quota exceeded"}, '{"error": "quota exceeded"}', "failed"),
    ],
    ids=["parts", "mapping", "error-mapping"],
)
async def test_audit_returns_the_answer_unchanged(
    tmp_path: Path, answer: Any, recorded: str, status: str
) -> None:
    auditor = JSONLToolAuditor(tmp_path / "audit.jsonl")

    async def handler(name: str, args: dict[str, Any]) -> Any:
        return answer

    wrapped = audit_tool_handler(handler, auditor, "a")

    assert await wrapped("tool", {}) is answer
    assert (auditor.entries[0].result, auditor.entries[0].status) == (recorded, status)


@pytest.mark.parametrize(
    "declined",
    [ToolRefusedError("not allowed here"), UnservedToolCallError("not mine")],
    ids=["refused", "unserved"],
)
async def test_audit_records_a_declined_call_as_failed_and_lets_it_through(
    tmp_path: Path, declined: Exception
) -> None:
    auditor = JSONLToolAuditor(tmp_path / "audit.jsonl")

    async def handler(name: str, args: dict[str, Any]) -> str:
        raise declined

    wrapped = audit_tool_handler(handler, auditor, "a")

    with pytest.raises(type(declined)):
        await wrapped("tool", {})
    assert (auditor.entries[0].status, auditor.entries[0].result) == ("failed", str(declined))


async def test_audit_records_a_cancelled_call(tmp_path: Path) -> None:
    auditor = JSONLToolAuditor(tmp_path / "audit.jsonl")
    started = asyncio.Event()

    async def handler(name: str, args: dict[str, Any]) -> str:
        started.set()
        await asyncio.Event().wait()
        return "never"

    task = asyncio.create_task(audit_tool_handler(handler, auditor, "a")("tool", {}))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert auditor.entries[0].status == "cancelled"


async def test_audit_records_a_call_the_channel_bound_cut_as_cancelled(tmp_path: Path) -> None:
    """The bound cancels the handler from outside: the audit sees a
    cancellation, the channel reads the timeout (RFC §15.8.1, §21.6)."""
    auditor = JSONLToolAuditor(tmp_path / "audit.jsonl")

    async def handler(name: str, args: dict[str, Any]) -> str:
        await asyncio.Event().wait()
        return "never"

    audited = audit_tool_handler(handler, auditor, "a")

    with pytest.raises(ToolTimeoutError):
        await answer_within(0.01, "tool", audited("tool", {}))
    assert auditor.entries[0].status == "cancelled"
