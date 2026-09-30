"""A reentry pass reads the room once (RMK-331, RFC §10.1 steps 6 and 12).

Each response event of a non-streamed turn is committed in its own pass,
under a fresh room lock. The pass reads what the lock protects once, the
room's context, and its status gate and its source's right to write read
that context rather than the store again.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from tests.test_framework import SimpleChannel

# The reads of the reentry pass running, by store method.
_PASS: ContextVar[dict[str, int] | None] = ContextVar("_PASS", default=None)


class _CountingStore(InMemoryStore):
    """Counts the room and binding reads made inside a reentry pass."""

    async def get_room(self, room_id: str) -> Any:
        _count("get_room")
        return await super().get_room(room_id)

    async def get_binding(self, room_id: str, channel_id: str) -> Any:
        _count("get_binding")
        return await super().get_binding(room_id, channel_id)


def _count(method: str) -> None:
    reads = _PASS.get()
    if reads is not None:
        reads[method] = reads.get(method, 0) + 1


def _count_passes(kit: RoomKit) -> list[dict[str, int]]:
    passes: list[dict[str, int]] = []
    run_pass = kit._run_reentry_pass

    async def counting(*args: Any, **kwargs: Any) -> None:
        reads: dict[str, int] = {}
        passes.append(reads)
        token = _PASS.set(reads)
        try:
            await run_pass(*args, **kwargs)
        finally:
            _PASS.reset(token)

    kit._run_reentry_pass = counting  # type: ignore[method-assign]
    return passes


def _answers(rounds: int) -> list[AIResponse]:
    steps = [
        AIResponse(
            content=f"Step {i}.",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id=f"t{i}", name="lookup", arguments={})],
        )
        for i in range(rounds)
    ]
    return [*steps, AIResponse(content="Answer.", finish_reason="stop")]


async def test_each_reentry_pass_reads_the_room_once_and_no_binding() -> None:
    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        return "ok"

    kit = RoomKit(store=_CountingStore())
    kit.register_channel(SimpleChannel("sms1"))
    provider = MockAIProvider(ai_responses=_answers(3))
    tools = [AITool(name="lookup", description="d")]
    kit.register_channel(AIChannel("ai1", provider=provider, tool_handler=lookup, tools=tools))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    passes = _count_passes(kit)

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )

    assert passes, "the turn's responses are committed in reentry passes"
    assert all(reads == {"get_room": 1} for reads in passes), passes
    await kit.close()
