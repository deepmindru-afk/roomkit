"""ON_ERROR fires once for a delegated turn that failed, whichever path its
delegation took (RMK-479, RFC §23.3 step 6).

A worker whose provider fails after a tool round: with a transport shared
into the child room its turn takes the room's path, without one the trace's;
each fires ON_ERROR once in the child room, as a room turn does in its room.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel

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


@pytest.mark.parametrize("shared", [False, True], ids=["trace-path", "transport-shared"])
async def test_a_failed_delegated_turn_fires_on_error_once(streaming: bool, shared: bool) -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    worker = Agent(
        "worker",
        provider=_FailsAfterARound(streaming=streaming),
        tools=[_LOOKUP],
        tool_handler=_found,
        tool_search=False,
    )
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
