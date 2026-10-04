"""ON_ERROR fires once for a delegated turn that failed, whichever path its
delegation took (RMK-479, RFC §23.3 step 6).

With a transport shared into the child room the turn takes the room's path,
without one the trace's; each fires ON_ERROR once in the child room, as a room
turn does in its room. Whatever shape the failure takes: an AIChannel worker
whose stream fails after a tool round (its provider streaming or not, one loop
either way), or a channel that answers then fails (a buffered reply with its
error), returns its error without answering, or raises.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import AILikeChannel, SimpleChannel

_LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})


class _FailsAfterARound(MockAIProvider):
    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        if len(self.calls) % 2:
            return AIResponse(
                content="Still checking.",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
            )
        raise ProviderError("upstream 400", provider="mock", status_code=400)


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


class _FailsBuffered(AILikeChannel):
    """Fails without a stream: answers then fails (``answers``), returns its
    error without answering (``returns``), or raises (``raises``)."""

    def __init__(self, channel_id: str, shape: str) -> None:
        super().__init__(channel_id)
        self._shape = shape

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        failure = ProviderError("upstream 400", provider="mock", status_code=400)
        if self._shape == "raises":
            raise failure
        if self._shape == "returns":
            return ChannelOutput(responded=False, error=failure)
        partial = RoomEvent(
            room_id=event.room_id,
            source=EventSource(channel_id=self.channel_id, channel_type=ChannelType.AI),
            content=TextContent(body="Partial work."),
            chain_depth=event.chain_depth + 1,
            metadata={"loop_end_reason": "error"},
        )
        return ChannelOutput(responded=True, response_events=[partial], error=failure)


PATHS = pytest.mark.parametrize("shared", [False, True], ids=["trace-path", "transport-shared"])


@PATHS
async def test_a_failed_delegated_stream_fires_on_error_once(
    streaming: bool, shared: bool
) -> None:
    worker = Agent(
        "worker",
        provider=_FailsAfterARound(streaming=streaming),
        tools=[_LOOKUP],
        tool_handler=_found,
        tool_search=False,
    )
    await _fires_on_error_once(worker, shared)


@PATHS
@pytest.mark.parametrize("shape", ["answers", "returns", "raises"])
async def test_a_failed_delegated_reply_fires_on_error_once(shape: str, shared: bool) -> None:
    await _fires_on_error_once(_FailsBuffered("worker", shape), shared)


async def _fires_on_error_once(worker: Channel, shared: bool) -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(worker)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    errors: list[tuple[str, str]] = []

    @kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC)
    async def _error(event: Any, ctx: Any) -> None:
        errors.append((event.room_id, event.source.channel_id))

    task = await kit.delegate(
        "r", "worker", "Find it.", wait=True, share_channels=["sms"] if shared else None
    )
    await asyncio.sleep(0.05)
    await kit.close()

    assert errors == [(task.child_room_id, "worker")]
