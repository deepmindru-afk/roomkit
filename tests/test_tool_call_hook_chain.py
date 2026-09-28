"""ON_TOOL_CALL's SYNC hooks chain on one outcome (RFC §9.3, RMK-273).

Each hook sees the result as the previous one left it, whichever way it
replaced it (``modify`` or ``metadata["result"]``); the model reads what the
last one left; the ASYNC observers see that final outcome, and still fire
after a BLOCK. A hook that de-tokenises arguments keeps the real values out
of the next turn's prompt.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

from roomkit import (
    Agent,
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
)
from roomkit.models.context import RoomContext
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider
from tests.test_framework import SimpleChannel
from tests.test_realtime_fixed_tools import call
from tests.test_realtime_skill_delivery import running
from tests.test_realtime_skills import _registry_with_skill

_LOOKUP = AITool(
    name="lookup",
    description="Look a customer up",
    parameters={"type": "object", "properties": {"email": {"type": "string"}}},
)


async def _crm(name: str, arguments: dict[str, Any]) -> str:
    return "SSN 123-45-6789"


async def _kit(streaming: bool, *answers: AIResponse) -> tuple[RoomKit, MockAIProvider]:
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"email": "<EMAIL_1>"})],
            ),
            *(answers or (AIResponse(content="done", finish_reason="stop"),)),
        ],
        streaming=streaming,
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(Agent("ai1", provider=provider, tool_handler=_crm, tools=[_LOOKUP]))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    return kit, provider


async def _say(kit: RoomKit, body: str = "who is it?") -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body=body))
    )
    await asyncio.sleep(0.05)  # the ASYNC observers are fire-and-forget


def _read_by_model(context: AIContext) -> list[str]:
    return [
        str(part.result)
        for message in context.messages
        if message.role == "tool"
        for part in message.content
    ]


def _redact(event: ToolCallEvent) -> ToolCallEvent:
    return dataclasses.replace(
        event, result=str(event.result).replace("123-45-6789", "[REDACTED]")
    )


async def test_a_modify_that_redacts_reaches_the_model_redacted(streaming: bool) -> None:
    kit, provider = await _kit(streaming)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="redact")
    async def redact(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.modify(_redact(event))

    await _say(kit)

    assert _read_by_model(provider.calls[1]) == ["SSN [REDACTED]"]
    await kit.close()


async def test_sync_hooks_chain_and_observers_see_the_final_result(streaming: bool) -> None:
    kit, provider = await _kit(streaming)
    observed: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="redact", priority=1)
    async def redact(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult(action="allow", metadata={"result": _redact(event).result})

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="cite", priority=2)
    async def cite(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.modify(
            dataclasses.replace(event, result=f"{event.result} (source: CRM)")
        )

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        observed.append(event.result)

    await _say(kit)

    assert _read_by_model(provider.calls[1]) == ["SSN [REDACTED] (source: CRM)"]
    assert observed == ["SSN [REDACTED] (source: CRM)"]
    await kit.close()


async def test_observers_fire_after_a_block_with_the_failure(streaming: bool) -> None:
    kit, provider = await _kit(streaming)
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="withhold")
    async def withhold(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.block("restricted")

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        observed.append(event)

    await _say(kit)

    assert len(observed) == 1
    assert observed[0].is_error
    assert "restricted" in str(observed[0].result)
    assert "123-45-6789" not in str(observed[0].result)
    await kit.close()


async def test_observers_fire_after_a_fail_closed_hook_fails(streaming: bool) -> None:
    kit, provider = await _kit(streaming)
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="pii", fail_closed=True)
    async def pii(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        raise RuntimeError("classifier down")

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        observed.append(event)

    await _say(kit)

    assert [event.is_error for event in observed] == [True]
    assert "123-45-6789" not in _read_by_model(provider.calls[1])[0]
    await kit.close()


async def test_the_next_prompt_keeps_the_model_s_arguments(streaming: bool) -> None:
    """A de-tokenising BEFORE_TOOL_USE hook hands the tool the real address;
    the usage digest of the next turn must quote the token, not the address."""
    kit, provider = await _kit(
        streaming,
        AIResponse(content="done", finish_reason="stop"),
        AIResponse(content="again", finish_reason="stop"),
    )

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="detokenize")
    async def detokenize(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult(action="allow", metadata={"arguments": {"email": "alice@real.example"}})

    await _say(kit)
    await _say(kit, "and then?")

    prompt = provider.calls[-1].system_prompt or ""
    assert "lookup" in prompt  # the digest is there...
    assert "alice@real.example" not in prompt  # ...without the real address
    await kit.close()


async def test_realtime_modify_reaches_the_model_and_the_observers(tmp_path) -> None:
    registry = _registry_with_skill(tmp_path, body="Rules.")
    async with running(registry, provider=MockRealtimeProvider()) as ctx:
        channel, provider, session, handler = ctx
        handler.return_value = {"ssn": "123-45-6789"}
        observed: list[ToolCallEvent] = []

        @channel._framework.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC)
        async def redact(event: ToolCallEvent, context: RoomContext) -> HookResult:
            return HookResult.modify(_redact(event))

        @channel._framework.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
        async def audit(event: ToolCallEvent, context: RoomContext) -> None:
            observed.append(event)

        result = await call(channel, provider, session, "calendar", {"action": "read"})
        await asyncio.sleep(0.05)

        assert result == {"ssn": "[REDACTED]"}
        assert ["[REDACTED]" in str(event.result) for event in observed] == [True]


async def test_realtime_observers_fire_after_a_block(tmp_path) -> None:
    registry = _registry_with_skill(tmp_path, body="Rules.")
    async with running(registry, provider=MockRealtimeProvider()) as ctx:
        channel, provider, session, handler = ctx
        handler.return_value = {"ssn": "123-45-6789"}
        observed: list[ToolCallEvent] = []

        @channel._framework.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC)
        async def withhold(event: ToolCallEvent, context: RoomContext) -> HookResult:
            return HookResult.block("restricted")

        @channel._framework.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
        async def audit(event: ToolCallEvent, context: RoomContext) -> None:
            observed.append(event)

        result = await call(channel, provider, session, "calendar", {"action": "read"})
        await asyncio.sleep(0.05)

        assert result == {"error": "restricted"}
        assert [event.is_error for event in observed] == [True]
        assert "123-45-6789" not in str(observed[0].result)
