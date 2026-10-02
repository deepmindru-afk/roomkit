"""``AIChannel.active_turns``: every path that produces a turn is counted.

The property is what a caller retiring a channel object reads before
``close()``: a channel displaced from the registry by a rebuild may still be
producing a turn that captured it, and ``close()`` tears the provider down
under that turn. A tool loop is counted through the steering registry it
already joins; a text-only stream, which has no loop context, is counted from
its first consumption to the close of its generator.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from roomkit.channels.ai import AIChannel
from roomkit.memory.mock import MockMemoryProvider
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    AIToolCall,
    StreamTextDelta,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event


class _GatedProvider(MockAIProvider):
    """Holds every generation on ``gate`` so a test can look at the channel
    mid-turn, then lets it finish once the gate opens."""

    def __init__(self, *, streaming: bool, ai_responses: list[AIResponse] | None = None) -> None:
        super().__init__(ai_responses=ai_responses, streaming=streaming)
        self.gate = asyncio.Event()
        self.reached = asyncio.Event()

    async def generate(self, context: AIContext) -> AIResponse:
        self.reached.set()
        await self.gate.wait()
        return await super().generate(context)

    async def generate_structured_stream(self, context: AIContext) -> Any:
        yield StreamTextDelta(text="Working ")
        self.reached.set()
        await self.gate.wait()
        async for event in super().generate_structured_stream(context):
            yield event


def _binding(tools: list[dict[str, Any]] | None = None) -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="room-1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": tools} if tools else {},
    )


def _context() -> RoomContext:
    return RoomContext(room=Room(id="room-1"))


class _TextOnlyGatedProvider(_GatedProvider):
    """Streams text alone: the default structured stream reads ``generate_stream``."""

    @property
    def supports_structured_streaming(self) -> bool:
        return False

    def generate_structured_stream(self, context: AIContext) -> Any:
        return AIProvider.generate_structured_stream(self, context)

    async def generate_stream(self, context: AIContext) -> Any:
        yield "Working "
        self.reached.set()
        await self.gate.wait()
        yield "done"


class _FailingMidStreamProvider(_GatedProvider):
    async def generate_structured_stream(self, context: AIContext) -> Any:
        yield StreamTextDelta(text="Working ")
        self.reached.set()
        await self.gate.wait()
        raise RuntimeError("provider went away")


_SEARCH_TOOL = {
    "name": "search",
    "description": "Search",
    "parameters": {"type": "object", "properties": {}},
}


async def _drain(stream: Any) -> list[Any]:
    return [chunk async for chunk in stream]


class TestActiveTurns:
    async def test_idle_channel_counts_zero(self) -> None:
        ch = AIChannel("ai1", provider=MockAIProvider())
        assert ch.active_turns == 0
        assert ch.info["active_turns"] == 0

    async def test_text_stream_counts_from_first_consumption_to_its_end(self) -> None:
        provider = _GatedProvider(streaming=True)
        ch = AIChannel("ai1", provider=provider)

        output = await ch.on_event(make_event(body="hi"), _binding(), _context())
        # Handing the output back produces nothing yet: the generator runs
        # when its consumer iterates it.
        assert ch.active_turns == 0

        consumer = asyncio.create_task(_drain(output.response_stream))
        await asyncio.wait_for(provider.reached.wait(), timeout=5)
        assert ch.active_turns == 1
        assert ch.info["active_turns"] == 1

        provider.gate.set()
        chunks = await asyncio.wait_for(consumer, timeout=5)
        assert "".join(c for c in chunks if isinstance(c, str)) == "Working Hello from AI"
        assert ch.active_turns == 0

    async def test_streaming_tool_loop_counts_through_the_registry(self) -> None:
        provider = _GatedProvider(
            streaming=True,
            ai_responses=[
                AIResponse(content="", tool_calls=[AIToolCall(id="tc1", name="search")]),
                AIResponse(content="Done", tool_calls=[]),
            ],
        )
        ch = AIChannel("ai1", provider=provider, tool_handler=AsyncMock(return_value="ok"))

        output = await ch.on_event(make_event(body="hi"), _binding([_SEARCH_TOOL]), _context())
        consumer = asyncio.create_task(_drain(output.response_stream))
        await asyncio.wait_for(provider.reached.wait(), timeout=5)
        assert ch.active_turns == 1
        assert len(ch._active_loops) == 1

        provider.gate.set()
        await asyncio.wait_for(consumer, timeout=5)
        assert ch.active_turns == 0

    async def test_a_turn_read_through_generate_counts_while_generating(self) -> None:
        provider = _GatedProvider(streaming=False)
        ch = AIChannel("ai1", provider=provider)

        output = await ch.on_event(make_event(body="hi"), _binding(), _context())
        consumer = asyncio.create_task(_drain(output.response_stream))
        await asyncio.wait_for(provider.reached.wait(), timeout=5)
        assert ch.active_turns == 1

        provider.gate.set()
        await asyncio.wait_for(consumer, timeout=5)
        assert ch.active_turns == 0

    async def test_stream_cancelled_from_the_outside_returns_to_zero(self) -> None:
        provider = _GatedProvider(streaming=True)
        ch = AIChannel("ai1", provider=provider)

        output = await ch.on_event(make_event(body="hi"), _binding(), _context())
        consumer = asyncio.create_task(_drain(output.response_stream))
        await asyncio.wait_for(provider.reached.wait(), timeout=5)
        assert ch.active_turns == 1

        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        assert ch.active_turns == 0

    async def test_a_caller_that_waits_for_idle_closes_after_the_stream(self) -> None:
        """The scene the property exists for: retire the object, but not
        under its stream. Waiting for zero before ``close()`` lets every
        delta through and tears the memory down after the last one."""
        provider = _GatedProvider(streaming=True)
        memory = MockMemoryProvider()
        ch = AIChannel("ai1", provider=provider, memory=memory)

        output = await ch.on_event(make_event(body="hi"), _binding(), _context())
        consumer = asyncio.create_task(_drain(output.response_stream))
        await asyncio.wait_for(provider.reached.wait(), timeout=5)

        async def retire() -> None:
            while ch.active_turns:
                await asyncio.sleep(0.001)
            await ch.close()

        retiring = asyncio.create_task(retire())
        await asyncio.sleep(0.01)
        assert memory.closed is False

        provider.gate.set()
        chunks = await asyncio.wait_for(consumer, timeout=5)
        await asyncio.wait_for(retiring, timeout=5)
        assert "".join(c for c in chunks if isinstance(c, str)) == "Working Hello from AI"
        assert memory.closed is True

    async def test_a_provider_that_streams_text_alone_is_counted_too(self) -> None:
        provider = _TextOnlyGatedProvider(streaming=True)
        ch = AIChannel("ai1", provider=provider)

        output = await ch.on_event(make_event(body="hi"), _binding(), _context())
        consumer = asyncio.create_task(_drain(output.response_stream))
        await asyncio.wait_for(provider.reached.wait(), timeout=5)
        assert ch.active_turns == 1

        provider.gate.set()
        chunks = await asyncio.wait_for(consumer, timeout=5)
        assert "".join(c for c in chunks if isinstance(c, str)) == "Working done"
        assert ch.active_turns == 0

    async def test_a_provider_error_mid_stream_returns_to_zero(self) -> None:
        provider = _FailingMidStreamProvider(streaming=True)
        ch = AIChannel("ai1", provider=provider)

        output = await ch.on_event(make_event(body="hi"), _binding(), _context())
        consumer = asyncio.create_task(_drain(output.response_stream))
        await asyncio.wait_for(provider.reached.wait(), timeout=5)
        assert ch.active_turns == 1

        provider.gate.set()
        results = await asyncio.gather(consumer, return_exceptions=True)
        assert isinstance(results[0], Exception)
        assert ch.active_turns == 0

    async def test_a_turn_without_tools_and_one_with_add_up(self) -> None:
        """Every turn runs a loop, with tools or without, and each counts."""
        provider = _GatedProvider(
            streaming=True,
            ai_responses=[
                AIResponse(content="", tool_calls=[AIToolCall(id="tc1", name="search")]),
                AIResponse(content="Done", tool_calls=[]),
            ],
        )
        ch = AIChannel("ai1", provider=provider, tool_handler=AsyncMock(return_value="ok"))

        text = await ch.on_event(make_event(body="hi"), _binding(), _context())
        loop = await ch.on_event(make_event(body="hi"), _binding([_SEARCH_TOOL]), _context())
        consumers = [
            asyncio.create_task(_drain(text.response_stream)),
            asyncio.create_task(_drain(loop.response_stream)),
        ]

        async def both_started() -> None:
            while ch.active_turns < 2:
                await asyncio.sleep(0)

        await asyncio.wait_for(both_started(), timeout=5)
        assert ch.active_turns == 2
        assert len(ch._active_loops) == 2

        provider.gate.set()
        await asyncio.wait_for(asyncio.gather(*consumers), timeout=5)
        assert ch.active_turns == 0
