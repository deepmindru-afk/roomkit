"""Execute one delegated turn in a child room and capture its result.

The single code path for running a delegated agent — used by both
``delegate(wait=True)`` (inline) and the background task runner via
:meth:`DelegationMixin`. Persists the worker's full trace (tool calls +
messages) into the child room and returns its output, either as free text or
as the structured payload of the result tool the delegation forces.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._tool_event_result import tool_event_result
from roomkit.core.mixins._result_capture import capture_result
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType, EventStatus, EventType
from roomkit.models.event import (
    EventSource,
    RoomEvent,
    TextContent,
    ToolCallContent,
    answer_text,
)
from roomkit.models.store_filter import EventFilter
from roomkit.models.streaming import LoopEndMarker, ToolCallEndMarker, ToolCallStartMarker

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.room import Room
    from roomkit.orchestration.result import ResultTool


_tasks_logger = logging.getLogger("roomkit.tasks")


def _child_tool_row(
    child_room_id: str,
    source: EventSource,
    marker: ToolCallStartMarker | ToolCallEndMarker,
    chain_depth: int,
) -> RoomEvent:
    """The TOOL_CALL_{START,END} row a streamed call marker leaves in the child room."""
    if isinstance(marker, ToolCallStartMarker):
        event_type = EventType.TOOL_CALL_START
        content = ToolCallContent(
            tool_name=marker.tool_name,
            tool_id=marker.tool_id,
            arguments=marker.arguments,
            status="pending",
        )
    else:
        event_type = EventType.TOOL_CALL_END
        content = ToolCallContent(
            tool_name=marker.tool_name,
            tool_id=marker.tool_id,
            arguments=marker.arguments,
            result=tool_event_result(marker.result),
            status=marker.status,
            duration_ms=marker.duration_ms,
            error=marker.error,
        )
    return RoomEvent(
        room_id=child_room_id,
        source=source,
        type=event_type,
        content=content,
        status=EventStatus.DELIVERED,
        chain_depth=chain_depth,
    )


async def _persist_child_stream(
    kit: RoomKit,
    child_room_id: str,
    sr: Any,
    chain_depth: int,
) -> str:
    """Drain a streaming response into the child room, persisting tool calls.

    Segments the stream the way the main inbound streaming path does: text
    deltas accumulate into MESSAGE segments split at tool-call boundaries,
    and each ``ToolCall{Start,End}Marker`` is persisted as a
    TOOL_CALL_{START,END} event — so the child room holds the worker's full
    trace (what it searched/ran, with arguments and results), not just its
    final answer. Unlike that path, the rows are committed directly: they
    ride no delivery lane and cross no BEFORE_BROADCAST hook, because nobody
    is delivered them — they are the record of a delegated turn. The turn's
    record (``loop_end_reason``, ``ai_usage``) rides its last message, as a
    non-streaming worker's does (RFC §6.4).
    Returns the last segment's text: the worker's answer, as a non-streaming
    worker's last message is.
    """
    source = EventSource(channel_id=sr.source_channel_id, channel_type=sr.source_channel_type)
    segment: list[str] = []
    record: dict[str, Any] = {}
    last: RoomEvent | None = None
    answer = ""

    async def _flush_segment() -> None:
        nonlocal last, answer
        if not segment:
            return
        body = answer = "".join(segment)
        segment.clear()
        last = await kit._commit_indexed(
            child_room_id,
            RoomEvent(
                room_id=child_room_id,
                source=source,
                type=EventType.MESSAGE,
                content=TextContent(body=body),
                status=EventStatus.DELIVERED,
                chain_depth=chain_depth,
                metadata=dict(record),
            ),
        )

    async for delta in sr.stream:
        if isinstance(delta, str):
            segment.append(delta)
        elif isinstance(delta, ToolCallStartMarker | ToolCallEndMarker):
            if isinstance(delta, ToolCallStartMarker):
                await _flush_segment()
            await kit._commit_indexed(
                child_room_id, _child_tool_row(child_room_id, source, delta, chain_depth)
            )
        elif isinstance(delta, LoopEndMarker):
            record = {"ai_usage": dict(delta.usage), "loop_end_reason": delta.reason}
        # ThinkingDeltaMarker (and any other marker): transient, not persisted.
    carried = bool(segment)
    await _flush_segment()
    if last is not None and record and not carried:
        # No final text: the record rides the last message already written
        await kit.store.update_event(
            last.model_copy(update={"metadata": {**last.metadata, **record}})
        )
    return answer


async def _persist_response_events(
    kit: RoomKit, child_room_id: str, response_events: list[RoomEvent]
) -> str | None:
    """Persist a non-streaming response's events (tool calls + messages) and
    return the last message text, so the child room keeps the full trace."""
    final_text: str | None = None
    for resp in response_events:
        await kit._commit_indexed(
            child_room_id, resp.model_copy(update={"status": EventStatus.DELIVERED})
        )
        # An interruption marker is not the worker's answer (RFC §6.4)
        if (text := answer_text(resp)) is not None:
            final_text = text
    return final_text


async def _child_context(kit: RoomKit, room: Room, bindings: list[ChannelBinding]) -> RoomContext:
    """The context a delegated turn reads, built after its message is stored.

    Built AFTER storing so the agent's memory provider can see the message in
    recent_events — which is the message just committed, so this read must be
    the room's tail (``newest_first``), not its head. A delegated room that
    outlives 50 events would otherwise hand the agent the opening turns and
    never the new one. Read whole, as ``_build_context`` reads it (RFC §7.5
    rule 8): the per-reader filter drops the refused rows for the channels
    that read it.
    """
    recent = await kit.store.list_events(
        room.id,
        offset=0,
        limit=50,
        newest_first=True,
        event_filter=EventFilter(include_blocked=True),
    )
    return RoomContext(room=room, bindings=bindings, recent_events=recent)


async def _broadcast_and_collect(
    kit: RoomKit, child_room_id: str, message_body: str
) -> str | None:
    """One delegated turn: store *message_body* as a system message, broadcast it,
    persist the agent's full trace (tool calls + messages), and return its text."""
    room = await kit.get_room(child_room_id)
    bindings = await kit.store.list_bindings(child_room_id)

    msg_event = RoomEvent(
        room_id=child_room_id,
        type=EventType.MESSAGE,
        source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
        content=TextContent(body=message_body),
        status=EventStatus.DELIVERED,
    )
    msg_event = await kit._commit_indexed(child_room_id, msg_event)
    context = await _child_context(kit, room, bindings)

    router = kit._get_router()
    source_binding = ChannelBinding(
        channel_id="system",
        room_id=child_room_id,
        channel_type=ChannelType.SYSTEM,
    )
    result = await router.broadcast(msg_event, source_binding, context)
    # What the broadcast blocked is stored and announced, and its side effects
    # kept, whichever path broadcast the trigger (RFC §8.3).
    await kit._commit_blocked_events(child_room_id, result)
    await kit._persist_side_effects(
        child_room_id, result.tasks, result.observations, msg_event, context
    )
    child_depth = msg_event.chain_depth + 1

    # Non-streaming: response_events already include the tool-call events —
    # persist them all (not just the final text) so the trace survives.
    for output in result.outputs.values():
        if not (output.responded and output.response_events):
            continue
        final_text = await _persist_response_events(kit, child_room_id, output.response_events)
        if output.error is not None:
            # A turn the provider interrupted after a round kept its trace and
            # has no answer: the delegated turn fails, as a streamed one does
            # (RFC §6.4).
            raise output.error
        if final_text is not None:
            return final_text

    # Streaming: drain the marker stream, persisting tool calls + text segments.
    for sr in result.streaming_responses:
        text = await _persist_child_stream(kit, child_room_id, sr, child_depth)
        if text:
            return text

    return None


#: A cursor larger than any real event index, so ``before_index`` returns the
#: tail (most recent events) — where a worker's final submit_result call lives.
#: Capped at max int32: the postgres store binds ``before_index`` as int4.
_LATEST_TAIL_CURSOR = 2**31 - 1


async def _scan_for_submitted_result(
    kit: RoomKit, child_room_id: str, result_tool: ResultTool | None = None
) -> dict[str, Any] | None:
    """Find a call of the result tool (``submit_result`` by default) in the
    worker's persisted trace.

    A call made through the channel's tool loop is caught by the capture
    handler; an agent whose tools come from an MCP server calls the tool there,
    out of that handler's reach, but the call is persisted as a TOOL_CALL event
    (named ``mcp__<server>__<name>``). Scanning the room's trace tail catches
    it. Returns the normalized payload, or None.
    """
    from roomkit.orchestration.result import SUBMIT_RESULT

    tool = result_tool or SUBMIT_RESULT
    events = await kit.store.list_events(
        child_room_id, before_index=_LATEST_TAIL_CURSOR, limit=100
    )
    for ev in reversed(events):
        if ev.type in (EventType.TOOL_CALL_END, EventType.TOOL_CALL_START):
            name = getattr(ev.content, "tool_name", "") or ""
            if tool.matches(name):
                return tool.normalize(getattr(ev.content, "arguments", None) or {})
    return None


@runtime_checkable
class _InjectableToolChannel(Protocol):
    """A channel that can host an injected tool and a swappable tool handler.

    ``_run_with_structured_result`` matches by structure, not by class, so
    duck-typed agent channels (and test doubles) qualify as long as they expose
    both members.
    """

    _room_tools: dict[str, list[Any]]
    tool_handler: Any


async def _run_with_structured_result(
    kit: RoomKit,
    child_room_id: str,
    task_desc: str,
    max_result_retries: int,
    result_tool: ResultTool | None = None,
) -> str:
    """Run a delegated agent that must hand its work back via a result tool
    (*result_tool*, ``submit_result`` by default). Injects the tool (for
    function-calling providers), runs the agent, and a deterministic completion
    guard: if the agent ends a turn without calling the tool, it is re-prompted to
    use it (up to *max_result_retries* times); if it still hasn't, the tool's
    ``on_missing`` payload is returned on its behalf.

    Capture is delivery-agnostic and scoped to *child_room_id*: a call made
    through the channel's tool loop is caught by the handler
    :func:`~roomkit.core.mixins._result_capture.capture_result` installs, and a
    call served by an MCP server instead is found in the room's persisted trace.
    Another room's delegation to the same agent never sees this one's result.
    Returns the payload as a JSON string (``on_missing``'s when exhausted)."""
    from roomkit.orchestration.result import SUBMIT_RESULT

    tool = result_tool or SUBMIT_RESULT
    room = await kit.get_room(child_room_id)
    agent_id = (room.metadata or {}).get("task_agent_id")
    channel = kit.channels.get(agent_id) if agent_id else None
    if not isinstance(channel, _InjectableToolChannel):
        # No injectable agent channel — fall back to plain text collection.
        text = await _broadcast_and_collect(kit, child_room_id, task_desc)
        return text or ""
    role = getattr(channel, "role", None) or getattr(channel, "description", None) or str(agent_id)

    with capture_result(channel, child_room_id, tool) as slot:
        message = task_desc
        last_text = ""
        for _attempt in range(max_result_retries + 1):
            text = await _broadcast_and_collect(kit, child_room_id, message)
            if slot.payload is not None:
                return json.dumps(slot.payload)
            scanned = await _scan_for_submitted_result(kit, child_room_id, tool)
            if scanned is not None:
                return json.dumps(scanned)
            last_text = text or last_text
            message = tool.reminder
        _tasks_logger.warning(
            "Delegated agent %s never called %s after %d attempts; failing.",
            agent_id,
            tool.name,
            max_result_retries + 1,
        )
        return json.dumps(
            tool.on_missing(role=role, last_output=last_text, attempts=max_result_retries + 1)
        )


async def run_agent_in_child_room(
    kit: RoomKit,
    child_room_id: str,
    task_desc: str,
    *,
    require_structured_result: bool = False,
    max_result_retries: int = 3,
    result_tool: ResultTool | None = None,
) -> str | None:
    """Send a task to a child room and collect the attached agent's response.

    This is the **single code path** for executing a delegated agent, used by
    both ``delegate(wait=True)`` (inline) and the background task runner.

    By default the agent's free-text response is collected and returned. When
    *require_structured_result* is set, the agent must instead hand its work back
    via a result tool, *result_tool* or ``submit_result`` by default (forced
    structure + a guaranteed result); the returned string is then the
    JSON-encoded structured payload (see :func:`_run_with_structured_result`).

    Either way the agent's full trace (tool calls + messages) is persisted in the
    child room, which records its parent via ``metadata.parent_room_id`` (set at
    creation in :meth:`delegate`) so the parent↔child link is rebuildable.
    """
    if require_structured_result:
        return await _run_with_structured_result(
            kit, child_room_id, task_desc, max_result_retries, result_tool
        )
    return await _broadcast_and_collect(kit, child_room_id, task_desc)
