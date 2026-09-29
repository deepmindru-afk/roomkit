"""Structured-result delegation: a delegated worker hands its work back via the
``submit_result`` tool, and a deterministic completion guard re-prompts (then
fails on the worker's behalf) if it never does.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from roomkit.core.mixins._child_execution import _scan_for_submitted_result
from roomkit.core.mixins.delegation import _run_with_structured_result
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.room import Room
from roomkit.orchestration.result import is_submit_result, normalize_result, orchestration_fail
from roomkit.orchestration.strategies.supervisor.prompts import SUBMIT_VERDICT


def _text_event(body: str) -> RoomEvent:
    return RoomEvent(
        room_id="parent::task-1",
        source=EventSource(channel_id="agent:w1", channel_type=ChannelType.AI),
        type=EventType.MESSAGE,
        content=TextContent(body=body),
    )


def _make_kit(
    agent_id: str,
    submit_on_attempt: int | None,
    payload: dict[str, Any] | None = None,
    tool_name: str = "submit_result",
):
    """Mock kit whose broadcast simulates the worker calling *tool_name* on the
    given 1-based attempt (None = never). Returns (kit, channel, counter), the
    counter holding the attempts and the message each one received."""
    kit = MagicMock()
    kit.get_room = AsyncMock(
        return_value=Room(id="parent::task-1", metadata={"task_agent_id": agent_id})
    )
    kit.store.list_bindings = AsyncMock(return_value=[])
    kit.store.list_events = AsyncMock(return_value=[])
    kit.store.add_event_auto_index = AsyncMock(side_effect=lambda _rid, ev: ev)
    kit.store.commit_event = AsyncMock(side_effect=lambda _rid, ev: ev)
    kit._commit_indexed = AsyncMock(side_effect=lambda _rid, ev: ev)

    channel = SimpleNamespace(_room_tools={}, tool_handler=None, role="Researcher")
    kit.channels = {agent_id: channel}
    counter: dict[str, Any] = {"n": 0, "messages": []}

    async def _broadcast(event, _binding, _context):
        counter["n"] += 1
        counter["messages"].append(getattr(event.content, "body", ""))
        if submit_on_attempt is not None and counter["n"] == submit_on_attempt:
            await channel.tool_handler(
                tool_name,
                payload or {"status": "completed", "summary": "done", "data": {"x": 1}},
            )
        out = SimpleNamespace(responded=True, response_events=[_text_event("raw text")])
        return SimpleNamespace(outputs={"w1": out}, streaming_responses=[])

    kit._get_router = MagicMock(
        return_value=SimpleNamespace(broadcast=AsyncMock(side_effect=_broadcast))
    )
    return kit, channel, counter


class TestStructuredResultGuard:
    async def test_returns_structured_payload_when_worker_submits(self) -> None:
        kit, channel, counter = _make_kit("agent:w1", submit_on_attempt=1)
        out = await _run_with_structured_result(
            kit, "parent::task-1", "do it", max_result_retries=3
        )
        payload = json.loads(out)
        assert payload["status"] == "completed"
        assert payload["data"] == {"x": 1}
        assert counter["n"] == 1  # no retries needed
        # The tool was declared in the child room only, and removed afterwards.
        assert channel._room_tools == {}
        assert channel.tool_handler is None

    async def test_reprompts_until_worker_submits(self) -> None:
        kit, _channel, counter = _make_kit("agent:w1", submit_on_attempt=3)
        out = await _run_with_structured_result(
            kit, "parent::task-1", "do it", max_result_retries=3
        )
        payload = json.loads(out)
        assert payload["status"] == "completed"
        assert counter["n"] == 3  # two nudges, submitted on the third turn

    async def test_orchestration_fail_when_never_submits(self) -> None:
        kit, _channel, counter = _make_kit("agent:w1", submit_on_attempt=None)
        out = await _run_with_structured_result(
            kit, "parent::task-1", "do it", max_result_retries=2
        )
        payload = json.loads(out)
        assert payload["status"] == "failed"
        assert payload["by"] == "orchestration"  # mechanism-level failure, not the worker's
        assert payload["reason"] == "no_structured_result_after_3_attempts"
        assert payload["role"] == "Researcher"
        assert payload["last_output"] == "raw text"  # explanatory context
        assert counter["n"] == 3  # initial + 2 retries


def _make_cc_kit(events: list[RoomEvent]):
    """Mock kit for a claude_code worker: it calls the gateway-exposed
    submit_result (so the wrapped tool_handler is NEVER invoked), and the call is
    persisted as a TOOL_CALL event that the scan must pick up."""
    kit = MagicMock()
    kit.get_room = AsyncMock(
        return_value=Room(id="parent::task-1", metadata={"task_agent_id": "agent:w1"})
    )
    kit.store.list_bindings = AsyncMock(return_value=[])
    kit.store.list_events = AsyncMock(return_value=events)
    kit.store.add_event_auto_index = AsyncMock(side_effect=lambda _rid, ev: ev)
    kit.store.commit_event = AsyncMock(side_effect=lambda _rid, ev: ev)
    kit._commit_indexed = AsyncMock(side_effect=lambda _rid, ev: ev)
    channel = SimpleNamespace(_room_tools={}, tool_handler=None, role="Researcher")
    kit.channels = {"agent:w1": channel}

    async def _broadcast(_event, _binding, _context):
        # The gateway handled submit_result; tool_handler is not called here.
        out = SimpleNamespace(responded=True, response_events=[_text_event("done")])
        return SimpleNamespace(outputs={"w1": out}, streaming_responses=[])

    kit._get_router = MagicMock(
        return_value=SimpleNamespace(broadcast=AsyncMock(side_effect=_broadcast))
    )
    return kit


def _tool_call_event(tool_name: str, arguments: dict[str, Any]) -> RoomEvent:
    return RoomEvent(
        room_id="parent::task-1",
        source=EventSource(channel_id="agent:w1", channel_type=ChannelType.AI),
        type=EventType.TOOL_CALL_END,
        content=ToolCallContent(
            tool_name=tool_name, tool_id="tc-1", arguments=arguments, status="completed"
        ),
    )


class TestStructuredResultViaTrace:
    """The claude_code path: submit_result arrives as a persisted tool call
    (prefixed by the gateway), captured by scanning the trace — not via the
    wrapped tool_handler."""

    async def test_captures_prefixed_gateway_call_from_trace(self) -> None:
        events = [
            _tool_call_event(
                "mcp__gateway__submit_result",
                {"status": "completed", "summary": "shipped", "data": {"y": 2}},
            )
        ]
        kit = _make_cc_kit(events)
        out = await _run_with_structured_result(
            kit, "parent::task-1", "do it", max_result_retries=3
        )
        payload = json.loads(out)
        assert payload["status"] == "completed"
        assert payload["summary"] == "shipped"
        assert payload["data"] == {"y": 2}

    async def test_fails_when_trace_has_no_submit_result(self) -> None:
        events = [_tool_call_event("mcp__gateway__web_search", {"q": "x"})]
        kit = _make_cc_kit(events)
        out = await _run_with_structured_result(
            kit, "parent::task-1", "do it", max_result_retries=1
        )
        payload = json.loads(out)
        assert payload["status"] == "failed"
        assert payload["by"] == "orchestration"


class TestResultHelpers:
    def test_is_submit_result_matches_bare_and_prefixed(self) -> None:
        assert is_submit_result("submit_result")
        assert is_submit_result("mcp__gateway__submit_result")
        assert not is_submit_result("web_search")
        assert not is_submit_result("submit_result_extra")

    def test_normalize_fills_defaults(self) -> None:
        r = normalize_result({"status": "completed", "summary": "s"})
        assert r == {
            "status": "completed",
            "summary": "s",
            "data": {},
            "deliverables": [],
            "reason": "",
        }

    def test_normalize_coerces_bad_types(self) -> None:
        r = normalize_result({"summary": "s", "data": "not-a-dict", "deliverables": "nope"})
        assert r["status"] == "completed"  # defaulted
        assert r["data"] == {}
        assert r["deliverables"] == []

    def test_orchestration_fail_shape(self) -> None:
        f = orchestration_fail(role="Analyst", last_output="partial", attempts=3)
        assert f["status"] == "failed"
        assert f["by"] == "orchestration"
        assert f["role"] == "Analyst"
        assert f["last_output"] == "partial"
        assert "3" in f["reason"]


class TestAnotherResultTool:
    """The guard forces whichever tool it is given; the supervised flow's verdict
    goes through ``submit_verdict`` this way (RMK-246)."""

    async def test_the_given_tool_is_injected_and_its_call_captured(self) -> None:
        kit, channel, _counter = _make_kit(
            "agent:boss",
            submit_on_attempt=1,
            payload={"approved": True, "feedback": "", "next_task": "write it up"},
            tool_name="submit_verdict",
        )
        seen: list[list[str]] = []
        original_broadcast = kit._get_router.return_value.broadcast.side_effect

        async def _spy(event, binding, context):
            seen.append([t.name for t in channel._room_tools.get("parent::task-1", [])])
            return await original_broadcast(event, binding, context)

        kit._get_router.return_value.broadcast.side_effect = _spy

        out = await _run_with_structured_result(
            kit, "parent::task-1", "judge it", max_result_retries=2, result_tool=SUBMIT_VERDICT
        )

        assert json.loads(out) == {"approved": True, "feedback": "", "next_task": "write it up"}
        assert seen == [["submit_verdict"]]
        assert channel._room_tools == {}

    async def test_its_reminder_and_its_missing_payload_are_used(self) -> None:
        kit, _channel, counter = _make_kit(
            "agent:boss", submit_on_attempt=None, tool_name="submit_verdict"
        )

        out = await _run_with_structured_result(
            kit, "parent::task-1", "judge it", max_result_retries=1, result_tool=SUBMIT_VERDICT
        )

        verdict = json.loads(out)
        assert verdict["approved"] is False
        assert verdict["next_task"] is None
        assert counter["messages"][1] == SUBMIT_VERDICT.reminder

    async def test_a_prefixed_gateway_call_is_found_in_the_trace(self) -> None:
        kit = _make_cc_kit(
            [
                _tool_call_event(
                    "mcp__gateway__submit_verdict",
                    {"approved": False, "feedback": "add sources", "next_task": ""},
                )
            ]
        )

        found = await _scan_for_submitted_result(kit, "parent::task-1", SUBMIT_VERDICT)

        assert found == {"approved": False, "feedback": "add sources", "next_task": None}
