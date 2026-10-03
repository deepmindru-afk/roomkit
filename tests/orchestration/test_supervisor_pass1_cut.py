"""A supervisor's task-formulation pass that hands on no task (RMK-436, RFC §19.7.3).

Cut short (its round cap), no worker runs and the message it answered still
gets an answer: the supervisor's fallback, stored and delivered with
``loop_end_reason`` naming the cut. Failed with a provider error, the caller
reads the error, logged once, with no fallback.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.core._fallback import FALLBACK_FAILED
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventType
from roomkit.models.event import TextContent
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel

LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})
LOOPING = AIResponse(
    content="Still checking.",
    finish_reason="tool_calls",
    tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
)


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


class _FailsAfterARound(MockAIProvider):
    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        if len(self.calls) % 2 == 1:
            return LOOPING
        raise ProviderError("upstream 400", provider="mock", status_code=400)


async def _supervised(provider: MockAIProvider) -> tuple[RoomKit, Any, SimpleChannel, list[str]]:
    kit = RoomKit()
    sms = SimpleChannel("sms")
    kit.register_channel(sms)
    supervisor = Agent(
        "sup",
        provider=provider,
        tools=[LOOKUP],
        tool_handler=_found,
        tool_search=False,
        max_tool_rounds=1,
    )
    worker = Agent("worker", provider=MockAIProvider(responses=["worker answer"]))
    kit.register_channel(supervisor)
    kit.register_channel(worker)
    delegated: list[str] = []

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC, name="delegated")
    async def on_delegated(event: Any, ctx: Any) -> None:
        delegated.append(event.metadata.get("agent_id"))

    orchestration = Supervisor(
        supervisor=supervisor, workers=[worker], strategy="parallel", auto_delegate=True
    )
    await kit.create_room(room_id="r", orchestration=orchestration)
    await kit.attach_channel("r", "sms")
    result = await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Find it."))
    )
    await asyncio.sleep(0.1)
    return kit, result, sms, delegated


def _delivered_messages(sms: SimpleChannel) -> list[str]:
    return [e.content.body for e in sms.delivered if e.type == EventType.MESSAGE]


async def test_a_cut_pass_answers_with_the_supervisors_fallback() -> None:
    kit, result, sms, delegated = await _supervised(
        MockAIProvider(ai_responses=[LOOPING] * 50, streaming=True)
    )

    assert delegated == []
    messages = [
        (e.content.body, e.metadata.get("loop_end_reason"))
        for e in await kit.get_timeline("r", limit=50)
        if e.source.channel_id == "sup" and e.type == EventType.MESSAGE
    ]
    assert messages == [(FALLBACK_FAILED, "max_rounds")]
    assert _delivered_messages(sms) == [FALLBACK_FAILED]
    assert result.error is None
    await kit.close()


async def test_a_failed_pass_gives_its_error_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="roomkit"):
        kit, result, sms, delegated = await _supervised(_FailsAfterARound(streaming=True))

    assert delegated == []
    assert isinstance(result.error, ProviderError)
    assert _delivered_messages(sms) == []
    # Named once, at DEBUG, as a room turn's error the caller receives.
    named = [r for r in caplog.records if "upstream 400" in r.getMessage()]
    assert [r.levelno for r in named] == [logging.DEBUG]
    await kit.close()
