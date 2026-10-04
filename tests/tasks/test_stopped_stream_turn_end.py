"""A response its transport stopped reading (a barge-in) ends ``cancelled``,
never read as an answer that completed (RMK-479, RFC §6.4, §12.2 step 13s).

Delegated with that transport shared, the worker's task fails as a turn a
stop cut does; as a room turn, the caller reads its end under ``turns``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, TaskStatus
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.base import AIContext, StreamEvent, StreamTextDelta
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


class _Slow(MockAIProvider):
    """Streams the start of an answer, then the rest slowly."""

    def __init__(self) -> None:
        super().__init__(streaming=True)

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        for word in ("The ", "partial "):
            yield StreamTextDelta(text=word)
        await asyncio.sleep(0.3)
        for word in ("rest ", "of the answer."):
            yield StreamTextDelta(text=word)


class _StopsReading(SimpleChannel):
    """Renders a stream and stops after two chunks, as a barge-in does."""

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        seen = 0
        async for chunk in text_stream:
            if isinstance(chunk, str):
                seen += 1
                if seen == 2:
                    return ChannelOutput.empty()
        return ChannelOutput.empty()


async def _kit() -> RoomKit:
    kit = RoomKit()
    kit.register_channel(_StopsReading("tx"))
    kit.register_channel(Agent("worker", provider=_Slow()))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    return kit


async def test_a_delegated_turn_its_transport_stopped_fails_cancelled() -> None:
    kit = await _kit()

    task = await kit.delegate("r", "worker", "Answer.", wait=True, share_channels=["tx"])
    await kit.close()

    assert task.result is not None
    assert task.result.status == TaskStatus.FAILED
    assert task.result.metadata.get("loop_end_reason") == "cancelled"


async def test_a_room_turn_its_transport_stopped_tells_its_caller_cancelled() -> None:
    kit = await _kit()
    await kit.attach_channel("r", "worker", category=ChannelCategory.INTELLIGENCE)

    result = await kit.process_inbound(
        InboundMessage(channel_id="tx", sender_id="u", content=TextContent(body="Answer."))
    )
    await kit.close()

    assert dict(result.response_metadata).get("turns") == {
        "worker": {"loop_end_reason": "cancelled"}
    }
