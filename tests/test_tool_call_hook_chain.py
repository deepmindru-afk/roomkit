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
    EventType,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
)
from roomkit.channels._tool_usage import ToolUsageMemory
from roomkit.models.context import RoomContext
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.external import PolicyExternalToolHandler
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


async def test_a_hook_that_clears_the_result_withholds_it(streaming: bool) -> None:
    """RMK-292: an empty replacement replaces (RFC §9.3); the original never
    reaches the model, the stored row or the observers."""
    kit, provider = await _kit(streaming)
    observed: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="clear")
    async def clear(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult(action="allow", metadata={"result": None})

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        observed.append(event.result)

    await _say(kit)

    assert _read_by_model(provider.calls[1]) == ["null"]
    assert observed == ["null"]
    ends = [e for e in await kit.store.list_events("r1") if e.type == EventType.TOOL_CALL_END]
    assert [e.content.result for e in ends] == ["null"]
    await kit.close()


async def test_observers_see_a_replacement_as_the_model_reads_it(streaming: bool) -> None:
    kit, provider = await _kit(streaming)
    observed: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="shape")
    async def shape(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.modify(dataclasses.replace(event, result={"n": 1}))

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        observed.append(event.result)

    await _say(kit)

    assert _read_by_model(provider.calls[1]) == ['{"n": 1}']
    assert observed == ['{"n": 1}']
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

    prompt = str(provider.calls[-1].messages[-1].content)  # the turn's notes
    assert "<EMAIL_1>" in prompt  # the digest quotes the model's arguments...
    assert "alice@real.example" not in prompt  # ...never the real address
    await kit.close()


async def test_a_cold_digest_keeps_the_model_s_arguments_too(streaming: bool) -> None:
    """After a restart the digest is rebuilt from the stored tool-call events;
    the end carries the arguments that ran, the start the model's request."""
    kit, provider = await _kit(
        streaming,
        AIResponse(content="done", finish_reason="stop"),
        AIResponse(content="again", finish_reason="stop"),
    )

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="detokenize")
    async def detokenize(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult(action="allow", metadata={"arguments": {"email": "alice@real.example"}})

    await _say(kit)
    agent = kit.channels["ai1"]
    agent._tool_usage = ToolUsageMemory()  # the process restarted: nothing in memory
    await _say(kit, "and then?")

    prompt = str(provider.calls[-1].messages[-1].content)  # the turn's notes
    assert "<EMAIL_1>" in prompt
    assert "alice@real.example" not in prompt
    await kit.close()


async def test_a_modify_the_chain_cannot_use_keeps_the_previous_rewrite(
    streaming: bool,
) -> None:
    """A MODIFY whose payload is no tool-call event replaces nothing (RFC
    §9.3): the chain carries on from the previous hook's redaction."""
    kit, provider = await _kit(streaming)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="redact", priority=1)
    async def redact(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult(action="allow", metadata={"result": _redact(event).result})

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="broken", priority=2)
    async def broken(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.modify(f"{event.result} (source: CRM)")  # a str, not the event

    await _say(kit)

    assert _read_by_model(provider.calls[1]) == ["SSN [REDACTED]"]
    await kit.close()


async def test_the_structured_copy_chains_too(streaming: bool) -> None:
    kit, provider = await _kit(streaming)
    seen: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="publish", priority=1)
    async def publish(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult(action="allow", metadata={"structured_content": {"rows": 1}})

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="read", priority=2)
    async def read(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        seen.append(event.structured_content)
        return HookResult.allow()

    await _say(kit)

    assert seen == [{"rows": 1}]
    await kit.close()


async def _external_report(
    sync_hook: Any,
) -> tuple[list[ToolCallEvent], PolicyExternalToolHandler, RoomKit]:
    kit = RoomKit()
    handler = PolicyExternalToolHandler()
    kit.register_channel(
        Agent("ext", provider=MockAIProvider(responses=["ok"]), external_tool_handler=handler)
    )
    await kit.create_room(room_id="r1")
    observed: list[ToolCallEvent] = []
    kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="sync")(sync_hook)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        observed.append(event)

    await handler._fire_on_tool_hook(
        "Read", {"path": "a.txt"}, "file contents", is_error=False, tool_call_id="t1", room_id="r1"
    )
    await asyncio.sleep(0.05)
    return observed, handler, kit


async def test_an_external_report_s_observers_see_the_provider_s_outcome() -> None:
    """An external tool already ran and its agent read the result: a hook's
    rewrite or block changes nothing there, and the audit must not record
    one either."""

    async def rewrite(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult(action="allow", metadata={"result": "REWRITTEN"})

    observed, _, kit = await _external_report(rewrite)
    assert [(e.result, e.is_error) for e in observed] == [("file contents", False)]
    await kit.close()

    async def block(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.block("restricted")

    observed, _, kit = await _external_report(block)
    assert [(e.result, e.is_error) for e in observed] == [("file contents", False)]
    await kit.close()

    async def modify(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.modify(dataclasses.replace(event, result="MODIFIED"))

    observed, _, kit = await _external_report(modify)
    assert [(e.result, e.is_error) for e in observed] == [("file contents", False)]
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


async def test_realtime_a_cleared_result_reads_null(tmp_path) -> None:
    """RMK-292: the realtime model reads what AIChannel reads for a cleared result."""
    registry = _registry_with_skill(tmp_path, body="Rules.")
    async with running(registry, provider=MockRealtimeProvider()) as ctx:
        channel, provider, session, handler = ctx
        handler.return_value = "SECRET-TOKEN-123"
        observed: list[ToolCallEvent] = []

        @channel._framework.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC)
        async def clear(event: ToolCallEvent, context: RoomContext) -> HookResult:
            return HookResult(action="allow", metadata={"result": None})

        @channel._framework.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
        async def audit(event: ToolCallEvent, context: RoomContext) -> None:
            observed.append(event)

        result = await call(channel, provider, session, "calendar", {"action": "read"})
        await asyncio.sleep(0.05)

        assert result is None  # the model read JSON null
        assert [event.result for event in observed] == ["null"]


async def test_realtime_observers_see_a_replacement_as_the_model_reads_it(tmp_path) -> None:
    registry = _registry_with_skill(tmp_path, body="Rules.")
    async with running(registry, provider=MockRealtimeProvider()) as ctx:
        channel, provider, session, handler = ctx
        observed: list[ToolCallEvent] = []

        @channel._framework.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC)
        async def shape(event: ToolCallEvent, context: RoomContext) -> HookResult:
            return HookResult.modify(dataclasses.replace(event, result={"n": 1}))

        @channel._framework.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
        async def audit(event: ToolCallEvent, context: RoomContext) -> None:
            observed.append(event)

        result = await call(channel, provider, session, "calendar", {"action": "read"})
        await asyncio.sleep(0.05)

        assert result == {"n": 1}
        assert [event.result for event in observed] == ['{"n": 1}']
