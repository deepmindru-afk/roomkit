"""run_agent_in_child_room persists the worker's FULL trace.

A delegated agent's child room must record what it actually did — each tool
call (with arguments and result) plus its text — not just the final answer, so
the room is a complete, linkable transcript. The parent link lives in the child
room's ``metadata.parent_room_id`` (asserted in test_integration), so the
parent↔child relationship is rebuildable from persistence alone.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.core.event_router import BroadcastResult
from roomkit.core.framework import RoomKit
from roomkit.core.mixins.delegation import _persist_child_stream, run_agent_in_child_room
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.room import Room
from roomkit.models.streaming import ThinkingDeltaMarker, ToolCallEndMarker, ToolCallStartMarker
from roomkit.providers.ai.base import AIImagePart, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.context import current_tool_call


def _recording_store() -> MagicMock:
    store = MagicMock()
    store.added = []

    async def _add(room_id: str, event: RoomEvent) -> RoomEvent:
        store.added.append(event)
        return event

    store.add_event_auto_index = AsyncMock(side_effect=_add)
    store.commit_event = AsyncMock(side_effect=_add)
    return store


def _sr(stream: Any) -> SimpleNamespace:
    return SimpleNamespace(
        stream=stream,
        source_channel_id="agent:w1",
        source_channel_type=ChannelType.AI,
        response_metadata={},
    )


def _recording_kit() -> MagicMock:
    kit = MagicMock()
    kit.store = _recording_store()
    kit._commit_indexed = kit.store.commit_event
    return kit


class TestPersistChildStream:
    async def test_persists_tool_calls_and_text_segments_in_order(self) -> None:
        kit = MagicMock()
        kit.store = _recording_store()
        kit._commit_indexed = kit.store.commit_event
        kit._commit_blocked_events = AsyncMock()
        kit._persist_side_effects = AsyncMock()

        async def _stream() -> Any:
            yield "Let me search. "
            yield ToolCallStartMarker(
                tool_name="WebSearch", tool_id="t1", arguments={"q": "world cup"}
            )
            yield ToolCallEndMarker(
                tool_name="WebSearch",
                tool_id="t1",
                arguments={"q": "world cup"},
                result="standings...",
                status="completed",
                duration_ms=42,
            )
            yield "Here is the answer."

        text = await _persist_child_stream(kit, "parent::task-1", _sr(_stream()), chain_depth=1)

        # Return value is the last segment, the worker's answer, as a
        # non-streaming worker's last message is (RMK-289).
        assert text == "Here is the answer."

        # Order: text segment, tool start, tool end, final text segment.
        seq = [(e.type, getattr(e.content, "tool_name", None)) for e in kit.store.added]
        assert seq == [
            (EventType.MESSAGE, None),
            (EventType.TOOL_CALL_START, "WebSearch"),
            (EventType.TOOL_CALL_END, "WebSearch"),
            (EventType.MESSAGE, None),
        ]

        # The tool-end event carries the arguments + result + timing.
        end = next(e for e in kit.store.added if e.type == EventType.TOOL_CALL_END)
        assert end.content.arguments == {"q": "world cup"}
        assert end.content.result == "standings..."
        assert end.content.duration_ms == 42
        assert end.content.status == "completed"

    async def test_a_tool_end_keeps_a_bounded_share_of_its_images(self) -> None:
        """The child room is persisted like any room: its TOOL_CALL_END events
        keep at most 512 KB of a result's images (RMK-260)."""
        kit = MagicMock()
        kit.store = _recording_store()
        kit._commit_indexed = kit.store.commit_event
        kit._commit_blocked_events = AsyncMock()
        kit._persist_side_effects = AsyncMock()
        header = "data:image/png;base64,"
        shot = AIImagePart(url=header + "A" * (300 * 1024 - len(header)), mime_type="image/png")

        async def _stream() -> Any:
            yield ToolCallStartMarker(tool_name="shoot", tool_id="t1", arguments={})
            yield ToolCallEndMarker(
                tool_name="shoot", tool_id="t1", result=[shot, shot, shot], status="completed"
            )

        await _persist_child_stream(kit, "parent::task-9", _sr(_stream()), chain_depth=1)

        end = next(e for e in kit.store.added if e.type == EventType.TOOL_CALL_END)
        assert sum(isinstance(p, AIImagePart) for p in end.content.result) == 1

    async def test_thinking_markers_are_not_persisted(self) -> None:
        kit = MagicMock()
        kit.store = _recording_store()
        kit._commit_indexed = kit.store.commit_event
        kit._commit_blocked_events = AsyncMock()
        kit._persist_side_effects = AsyncMock()

        async def _stream() -> Any:
            yield ThinkingDeltaMarker(thinking="hmm")
            yield "final answer"

        text = await _persist_child_stream(kit, "parent::task-2", _sr(_stream()), chain_depth=1)
        assert text == "final answer"
        # Only the text segment is persisted — thinking is transient.
        assert [e.type for e in kit.store.added] == [EventType.MESSAGE]

    async def test_text_only_stream_persists_single_message(self) -> None:
        kit = MagicMock()
        kit.store = _recording_store()
        kit._commit_indexed = kit.store.commit_event
        kit._commit_blocked_events = AsyncMock()
        kit._persist_side_effects = AsyncMock()

        async def _stream() -> Any:
            yield "just "
            yield "text"

        text = await _persist_child_stream(kit, "parent::task-3", _sr(_stream()), chain_depth=1)
        assert text == "just text"
        assert len(kit.store.added) == 1
        assert kit.store.added[0].content.body == "just text"


class TestAChildTraceCutShort:
    """RMK-291: a delegated turn cut short leaves no call open in its child room."""

    async def test_a_failed_stream_closes_its_open_call(self) -> None:
        kit = _recording_kit()

        async def _stream() -> Any:
            yield "Looking. "
            yield ToolCallStartMarker(tool_name="search", tool_id="t1", arguments={})
            raise RuntimeError("upstream 500")

        with pytest.raises(RuntimeError, match="upstream 500"):
            await _persist_child_stream(kit, "parent::task-4", _sr(_stream()), chain_depth=1)

        rows = [(e.type, e.content) for e in kit.store.added]
        assert [t for t, _ in rows] == [
            EventType.MESSAGE,
            EventType.TOOL_CALL_START,
            EventType.TOOL_CALL_END,
        ]
        end = rows[-1][1]
        assert (end.tool_id, end.status, end.error) == ("t1", "failed", "turn failed")

    async def test_a_delegation_cancelled_while_its_tool_runs_leaves_no_call_open(
        self, streaming: bool
    ) -> None:
        started = asyncio.Event()

        async def _slow(name: str, args: dict[str, Any]) -> str:
            started.set()
            await asyncio.sleep(3600)
            return "never"

        kit = _delegating_kit(
            MockAIProvider(
                streaming=streaming,
                ai_responses=[
                    AIResponse(
                        content="Working.",
                        finish_reason="tool_calls",
                        tool_calls=[AIToolCall(id="tc1", name="slow", arguments={})],
                    ),
                    AIResponse(content="Done."),
                ],
            ),
            _slow,
            "slow",
        )
        await kit.create_room(room_id="parent")
        task = asyncio.create_task(kit.delegate("parent", "worker", "go", wait=True))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # Every start has its end: a start row never stays pending (RFC §23.3).
        rows = await _child_rows(kit)
        starts = {c.tool_id for t, c in rows if t == EventType.TOOL_CALL_START}
        ends = {c.tool_id: c for t, c in rows if t == EventType.TOOL_CALL_END}
        assert starts == set(ends)
        assert all((c.status, c.error) == ("failed", "cancelled") for c in ends.values())
        if streaming:
            assert starts == {"tc1"}
            message = next(c for t, c in rows if t == EventType.MESSAGE)
            assert message.body == "Working."
        await kit.close()

    async def test_a_tool_end_keeps_its_structured_copy(self, streaming: bool) -> None:
        async def _query(name: str, args: dict[str, Any]) -> str:
            current_tool_call().structured_content = {"rows": [1, 2, 3]}
            return "3 rows"

        kit = _delegating_kit(
            MockAIProvider(
                streaming=streaming,
                ai_responses=[
                    AIResponse(
                        content="",
                        finish_reason="tool_calls",
                        tool_calls=[AIToolCall(id="tc1", name="query", arguments={})],
                    ),
                    AIResponse(content="Done."),
                ],
            ),
            _query,
            "query",
        )
        await kit.create_room(room_id="parent")

        await kit.delegate("parent", "worker", "go", wait=True)

        end = next(c for t, c in await _child_rows(kit) if t == EventType.TOOL_CALL_END)
        assert (end.result, end.structured_content) == ("3 rows", {"rows": [1, 2, 3]})
        await kit.close()


def _delegating_kit(provider: MockAIProvider, handler: Any, tool: str) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(
        AIChannel(
            "worker",
            provider=provider,
            tool_handler=handler,
            tools=[AITool(name=tool, description="d")],
            tool_search=False,
        )
    )
    return kit


async def _child_rows(kit: RoomKit) -> list[tuple[EventType, Any]]:
    (child,) = [r for r in await kit.store.list_rooms() if r.id != "parent"]
    return [
        (e.type, e.content)
        for e in await kit.store.list_events(child.id)
        if e.source.channel_id == "worker"
    ]


class TestRunAgentNonStreaming:
    async def test_persists_all_response_events_not_just_final_text(self) -> None:
        kit = MagicMock()
        kit.store = _recording_store()
        kit._commit_indexed = kit.store.commit_event
        kit._commit_blocked_events = AsyncMock()
        kit._persist_side_effects = AsyncMock()
        kit.get_room = AsyncMock(
            return_value=Room(id="parent::task-1", metadata={"parent_room_id": "parent"})
        )
        kit.store.list_bindings = AsyncMock(return_value=[])
        kit.store.list_events = AsyncMock(return_value=[])

        source = EventSource(channel_id="agent:w1", channel_type=ChannelType.AI)
        tool_event = RoomEvent(
            room_id="parent::task-1",
            source=source,
            type=EventType.TOOL_CALL_END,
            content=ToolCallContent(tool_name="WebSearch", tool_id="t1", status="completed"),
        )
        msg_event = RoomEvent(
            room_id="parent::task-1",
            source=source,
            type=EventType.MESSAGE,
            content=TextContent(body="the answer"),
        )
        output = SimpleNamespace(
            responded=True, error=None, response_events=[tool_event, msg_event]
        )
        broadcast_result = BroadcastResult(outputs={"w1": output})

        router = MagicMock()
        router.broadcast = AsyncMock(return_value=broadcast_result)
        kit._get_router = MagicMock(return_value=router)

        text = await run_agent_in_child_room(kit, "parent::task-1", "do the task")

        assert text == "the answer"
        # task message + tool-call event + final message all persisted.
        persisted_types = [e.type for e in kit.store.added]
        assert EventType.TOOL_CALL_END in persisted_types
        assert persisted_types.count(EventType.MESSAGE) == 2  # task + answer
