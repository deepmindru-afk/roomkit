"""Who serves a call is decided call by call (RFC §9.3, RMK-308).

A call the provider already ran is reported; a call to a tool the channel does
not serve is its external tool handler's when one is configured; every other
call is the channel's own, through the gate, then its handler, or unserved.
Every case runs against a provider that streams and one read through its
``generate()``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    EventType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import TextContent, ToolCallContent
from roomkit.models.hook import HookResult
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall, AIToolCallPart, ServedCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.external import ExternalToolHandler, ToolDecision
from tests.test_framework import SimpleChannel

LOOKUP = AITool(name="lookup", description="Look it up.", parameters={"type": "object"})


class _Proxy(ExternalToolHandler):
    """Approves every call it is asked about, or denies every one, and records
    what reaches it."""

    def __init__(self, *, approves: bool = True) -> None:
        super().__init__()
        self.approves = approves
        self.decided: list[str] = []
        self.results: list[str] = []
        self.refused: list[str] = []

    async def process_tool_call(
        self, tool_name: str, tool_input: dict[str, Any], **kwargs: Any
    ) -> ToolDecision:
        self.decided.append(tool_name)
        if not self.approves:
            return ToolDecision(approved=False, reason="not on this host")
        return ToolDecision(approved=True, result="proxy ran it")

    async def on_tool_result(
        self, tool_name: str, tool_input: dict[str, Any], result: str, **kwargs: Any
    ) -> None:
        self.results.append(tool_name)

    async def on_tool_refused(
        self, tool_name: str, tool_input: dict[str, Any], reason: str, **kwargs: Any
    ) -> None:
        self.refused.append(tool_name)
        await super().on_tool_refused(tool_name, tool_input, reason, **kwargs)


def _calls(*calls: AIToolCall) -> AIResponse:
    return AIResponse(content="Working.", finish_reason="tool_calls", tool_calls=list(calls))


class _Local:
    """The channel's own handler, recording the calls it serves."""

    def __init__(self) -> None:
        self.served: list[str] = []

    async def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        self.served.append(name)
        return "served here"


class _Room:
    def __init__(self, ai: AIChannel) -> None:
        self.kit = RoomKit()
        self.ai = ai
        self.observed: list[ToolCallEvent] = []

    async def open(self, **binding_metadata: Any) -> _Room:
        self.kit.register_channel(SimpleChannel("sms1"))
        self.kit.register_channel(self.ai)
        await self.kit.create_room(room_id="r1")
        await self.kit.attach_channel("r1", "sms1")
        await self.kit.attach_channel(
            "r1", "ai1", category=ChannelCategory.INTELLIGENCE, metadata=binding_metadata
        )

        @self.kit.hook(HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC, name="observe")
        async def observe(event: ToolCallEvent, ctx: Any) -> None:
            self.observed.append(event)

        return self

    async def say(self) -> None:
        await self.kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u", content=TextContent(body="go"))
        )
        await asyncio.sleep(0.05)

    async def ends(self) -> list[ToolCallContent]:
        events = await self.kit.store.list_events("r1")
        return [
            e.content
            for e in events
            if e.type == EventType.TOOL_CALL_END and isinstance(e.content, ToolCallContent)
        ]


async def test_a_call_the_provider_ran_is_reported_beside_local_tools(streaming: bool) -> None:
    """The provider's result rides the call; a local tool beside it does
    not make it the channel's to dispatch."""
    ran = AIToolCall(
        id="b1", name="Bash", arguments={"cmd": "ls"}, served=ServedCall(result="a.txt")
    )
    proxy, local = _Proxy(), _Local()
    provider = MockAIProvider(ai_responses=[_calls(ran)], streaming=streaming)
    ai = AIChannel(
        "ai1", provider=provider, external_tool_handler=proxy, tools=[LOOKUP], tool_handler=local
    )
    room = await _Room(ai).open()

    await room.say()

    assert proxy.results == ["Bash"] and proxy.decided == []
    assert local.served == []
    [end] = await room.ends()
    assert end.arguments == {"cmd": "ls"}
    assert end.status == "completed"


async def test_a_call_the_provider_ran_reaches_the_observers_without_a_handler(
    streaming: bool,
) -> None:
    """With no external handler, the provider's own call still gets its
    rows and its report, whatever the provider streams."""
    ran = AIToolCall(
        id="b1", name="Bash", arguments={"cmd": "ls"}, served=ServedCall(result="a.txt")
    )
    provider = MockAIProvider(ai_responses=[_calls(ran)], streaming=streaming)
    room = await _Room(AIChannel("ai1", provider=provider)).open()

    await room.say()

    [end] = await room.ends()
    assert end.tool_name == "Bash" and end.status == "completed"
    assert [(e.name, e.is_error) for e in room.observed] == [("Bash", False)]


async def test_the_handler_decides_a_tool_the_channel_does_not_serve(streaming: bool) -> None:
    """A pending call to a provider-side tool is the external handler's; one
    to the channel's own tool is the channel's, handler or not."""
    provider = MockAIProvider(
        ai_responses=[
            _calls(
                AIToolCall(id="p1", name="Bash", arguments={"cmd": "ls"}),
                AIToolCall(id="l1", name="lookup", arguments={}),
            ),
            AIResponse(content="Done."),
        ],
        streaming=streaming,
    )
    proxy, local = _Proxy(), _Local()
    ai = AIChannel(
        "ai1", provider=provider, external_tool_handler=proxy, tools=[LOOKUP], tool_handler=local
    )
    room = await _Room(ai).open()

    await room.say()

    assert proxy.decided == ["Bash"]
    assert local.served == ["lookup"]
    # The next round reads every call of the round, the provider's included,
    # each with its result.
    replay = provider.calls[-1].messages
    asked = next(m for m in replay if m.role == "assistant" and isinstance(m.content, list))
    assert [p.name for p in asked.content if isinstance(p, AIToolCallPart)] == ["Bash", "lookup"]
    answered = next(m for m in replay if m.role == "tool")
    assert [(p.name, p.result) for p in answered.content] == [
        ("Bash", "proxy ran it"),
        ("lookup", "served here"),
    ]


async def test_a_block_refuses_a_call_on_a_channel_without_a_handler(streaming: bool) -> None:
    """BEFORE_TOOL_USE's BLOCK is the call's refusal, which the model
    reads, and the turn goes on."""
    provider = MockAIProvider(
        ai_responses=[
            _calls(AIToolCall(id="p1", name="Bash", arguments={"cmd": "rm -rf /"})),
            AIResponse(content="Understood."),
        ],
        streaming=streaming,
    )
    room = _Room(AIChannel("ai1", provider=provider))
    await room.open(tools=[{"name": "Bash", "description": "b", "parameters": {"type": "object"}}])

    @room.kit.hook(HookTrigger.BEFORE_TOOL_USE, name="deny")
    async def deny(event: ToolCallEvent, ctx: Any) -> HookResult:
        return HookResult.block("denied by policy")

    await room.say()

    [end] = await room.ends()
    assert end.status == "failed"
    assert [(e.name, e.is_error) for e in room.observed] == [("Bash", True)]
    tool_message = next(m for m in provider.calls[-1].messages if m.role == "tool")
    assert "denied by policy" in str(tool_message.content)
    assert len(provider.calls) == 2


async def test_a_tool_the_hook_withdrew_stays_the_channels_to_refuse(streaming: bool) -> None:
    """A tool BEFORE_AI_GENERATION withdrew is gone from the turn: with an
    external handler at hand too, the channel's gate refuses it."""
    provider = MockAIProvider(
        ai_responses=[
            _calls(AIToolCall(id="l1", name="lookup", arguments={})),
            AIResponse(content="ok"),
        ],
        streaming=streaming,
    )
    proxy, local = _Proxy(), _Local()
    ai = AIChannel(
        "ai1", provider=provider, external_tool_handler=proxy, tools=[LOOKUP], tool_handler=local
    )
    room = await _Room(ai).open()

    @room.kit.hook(HookTrigger.BEFORE_AI_GENERATION, name="withdraw")
    async def withdraw(event: Any, ctx: Any) -> HookResult:
        event.ai_context.tools[:] = [t for t in event.ai_context.tools if t.name != "lookup"]
        return HookResult.allow()

    await room.say()

    assert proxy.decided == [] and local.served == []
    [end] = await room.ends()
    assert end.outcome == "refused"


async def test_a_call_the_handler_denies_is_a_refusal(streaming: bool) -> None:
    """The handler's denial is a refusal, stored as the channel's own
    refusals are; the provider's loop, not the channel's, reads it."""
    provider = MockAIProvider(
        ai_responses=[
            _calls(AIToolCall(id="p1", name="Bash", arguments={"cmd": "rm -rf /"})),
            AIResponse(content="Understood."),
        ],
        streaming=streaming,
    )
    proxy = _Proxy(approves=False)
    room = await _Room(AIChannel("ai1", provider=provider, external_tool_handler=proxy)).open()

    await room.say()

    [end] = await room.ends()
    assert end.outcome == "refused"
    assert "not on this host" in str(end.result)
    # Its own refusal comes back to it as one (RMK-432).
    assert (proxy.refused, proxy.results) == (["Bash"], [])


async def test_a_call_written_unreadable_is_refused_before_the_handler(streaming: bool) -> None:
    """A provider-side call whose arguments do not read never reaches the
    external handler, and the refusal says the model wrote them so."""
    garbled = AIToolCall(
        id="p1", name="Bash", arguments={"raw": "[1, 2]"}, partial=True, garbled=True
    )
    provider = MockAIProvider(ai_responses=[_calls(garbled)], streaming=streaming)
    proxy = _Proxy()
    room = await _Room(AIChannel("ai1", provider=provider, external_tool_handler=proxy)).open()

    await room.say()

    assert proxy.decided == []
    [end] = await room.ends()
    assert end.outcome == "refused"
    assert "arguments unreadable" in str(end.result)
