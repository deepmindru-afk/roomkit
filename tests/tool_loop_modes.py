"""Drive an AIChannel tool loop the same way in both generation modes.

Every in-repo provider streams, so an AIChannel with tools runs the streaming
tool loop in production; the non-streaming loop serves a provider that cannot
stream. A tool-loop test takes the ``streaming`` fixture (``tests/conftest.py``)
and drives the turn through the helpers below, which read either loop's
outcome into the same :class:`LoopRun`:

- :func:`respond` goes through ``on_event``, where the provider picks the
  loop: pass the fixture's value to ``MockAIProvider(streaming=...)``. Under
  the fixture it checks that the reply came in the mode the test runs in.
- :func:`run_tool_loop` calls a loop directly, picked by its ``streaming``
  argument; the provider's flag plays no part.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import EventType
from roomkit.models.event import RoomEvent, TextContent, ToolCallContent
from roomkit.models.streaming import (
    LoopEndMarker,
    LoopEndReason,
    ToolCallEndMarker,
    ToolCallStartMarker,
)
from roomkit.providers.ai.base import AIContext

# The mode the running test was parametrised with, set by the ``streaming``
# fixture for the length of one test; ``None`` outside it.
_expected_streaming: bool | None = None


def expect_streaming(value: bool | None) -> None:
    """Record the mode the running test runs in (the ``streaming`` fixture)."""
    global _expected_streaming
    _expected_streaming = value


@dataclass
class LoopCall:
    """One tool call the loop ran, as it reported it.

    ``requested`` is what the model asked for (the call's start); ``arguments``
    is what the loop reports as executed (the call's end), after folds and a
    ``BEFORE_TOOL_USE`` rewrite.
    """

    name: str
    id: str
    requested: dict[str, Any]
    arguments: dict[str, Any]
    result: Any
    failed: bool
    error: str | None = None
    structured_content: dict[str, Any] | None = None


@dataclass
class LoopRun:
    """What one turn of a tool loop produced, whichever loop ran it.

    ``said`` is every non-empty text segment in order: what the model said
    before each tool round, then its answer. ``text`` is the answer alone.
    ``reason`` is the loop's end reason (``LoopEndMarker.reason`` /
    ``ToolLoopResult.reason``); ``None`` when a non-streaming reply carries no
    final message to read it from. ``metadata`` is the reply's response
    metadata (the turn's live record), read once the turn has ended.
    """

    text: str = ""
    said: list[str] = field(default_factory=list)
    calls: list[LoopCall] = field(default_factory=list)
    reason: LoopEndReason | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def open_tool_calls(events: list[RoomEvent]) -> list[str]:
    """Ids of the stored TOOL_CALL_START events that no TOOL_CALL_END closes."""
    ended = {
        e.content.tool_id
        for e in events
        if e.type == EventType.TOOL_CALL_END and isinstance(e.content, ToolCallContent)
    }
    return [
        e.content.tool_id
        for e in events
        if e.type == EventType.TOOL_CALL_START
        and isinstance(e.content, ToolCallContent)
        and e.content.tool_id not in ended
    ]


async def run_tool_loop(channel: AIChannel, context: AIContext, *, streaming: bool) -> LoopRun:
    """Run the channel's tool loop for ``context`` in the given mode."""
    if streaming:
        return await _read_stream(channel._run_streaming_tool_loop(context))
    result = await channel._run_tool_loop(context)
    run = LoopRun(text=result.response.content or "", reason=result.reason)
    for rnd in result.rounds:
        if rnd.text_before:
            run.said.append(rnd.text_before)
        for call, part in zip(rnd.tool_calls, rnd.results, strict=False):
            run.calls.append(
                LoopCall(
                    name=call.name,
                    id=call.id,
                    requested=call.arguments,
                    arguments=rnd.arguments_ran(call),
                    result=part.result,
                    failed=part.is_error,
                    error=part.as_text() if part.is_error else None,
                    structured_content=part.structured_content,
                )
            )
    if run.text:
        run.said.append(run.text)
    return run


async def respond(
    channel: AIChannel, event: RoomEvent, binding: ChannelBinding, context: RoomContext
) -> LoopRun:
    """Deliver ``event`` to the channel and read its reply, streamed or not."""
    output = await channel.on_event(event, binding, context)
    streamed = output.response_stream is not None
    if _expected_streaming is not None and streamed != _expected_streaming:
        raise AssertionError(
            f"the test runs streaming={_expected_streaming} but the reply came "
            f"streaming={streamed}: pass the fixture to the provider"
        )
    if output.response_stream is not None:
        run = await _read_stream(output.response_stream)
    else:
        run = _read_events(output.response_events)
    run.metadata = dict(output.response_metadata or {})
    return run


async def _read_stream(stream: AsyncIterator[Any]) -> LoopRun:
    run = LoopRun()
    pending: list[str] = []
    requested: dict[str, dict[str, Any]] = {}
    in_round = False
    async for delta in stream:
        if isinstance(delta, str):
            pending.append(delta)
            in_round = False
        elif isinstance(delta, ToolCallStartMarker):
            if not in_round:
                _flush(run, pending)
                in_round = True
            requested[delta.tool_id] = delta.arguments
        elif isinstance(delta, ToolCallEndMarker):
            in_round = False
            run.calls.append(
                LoopCall(
                    name=delta.tool_name,
                    id=delta.tool_id,
                    requested=requested.get(delta.tool_id, {}),
                    arguments=delta.arguments,
                    result=delta.result,
                    failed=delta.status == "failed",
                    error=delta.error,
                    structured_content=delta.structured_content,
                )
            )
        elif isinstance(delta, LoopEndMarker):
            run.reason = delta.reason
    run.text = "".join(pending)
    _flush(run, pending)
    return run


def _flush(run: LoopRun, pending: list[str]) -> None:
    text = "".join(pending)
    pending.clear()
    if text:
        run.said.append(text)


def _read_events(events: list[RoomEvent]) -> LoopRun:
    run = LoopRun()
    requested: dict[str, dict[str, Any]] = {}
    for event in events:
        content = event.content
        if event.type == EventType.MESSAGE and isinstance(content, TextContent):
            if content.body:
                run.said.append(content.body)
            if "loop_end_reason" in event.metadata:
                run.text = content.body
                run.reason = event.metadata["loop_end_reason"]
        elif event.type == EventType.TOOL_CALL_START and isinstance(content, ToolCallContent):
            requested[content.tool_id] = content.arguments
        elif event.type == EventType.TOOL_CALL_END and isinstance(content, ToolCallContent):
            run.calls.append(
                LoopCall(
                    name=content.tool_name,
                    id=content.tool_id,
                    requested=requested.get(content.tool_id, {}),
                    arguments=content.arguments,
                    result=content.result,
                    failed=content.status == "failed",
                    error=content.error,
                    structured_content=content.structured_content,
                )
            )
    return run
