"""Streamed AI segments must pass through BEFORE_BROADCAST sync hooks.

The streaming path used to persist and re-broadcast segments without ever
running BEFORE_BROADCAST (only AFTER_BROADCAST fired). A pre-broadcast guard
on AI output — e.g. PII de-anonymisation — was therefore bypassed in
streaming: the stored row kept the raw text. These tests pin that a hook's
modification lands on the persisted segment and that a block drops it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Literal

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.event_router import StreamingResponse
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.store_filter import PersistencePolicy
from roomkit.models.task import Observation, Task
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


async def _wire(kit: RoomKit, content: str) -> None:
    provider = MockAIProvider(
        streaming=True,
        ai_responses=[AIResponse(content=content, finish_reason="stop")],
    )
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(AIChannel("ai1", provider=provider))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)


def _ai_messages(events: list[RoomEvent]) -> list[RoomEvent]:
    return [e for e in events if e.type == EventType.MESSAGE and e.source.channel_id == "ai1"]


_TOOL_EVENTS = (EventType.TOOL_CALL_START, EventType.TOOL_CALL_END)


def _tool_events(events: list[RoomEvent]) -> list[RoomEvent]:
    return [e for e in events if e.type in _TOOL_EVENTS]


async def _wire_tool_turn(kit: RoomKit) -> SimpleChannel:
    """A streamed turn that calls one tool, then answers; returns the transport."""
    provider = MockAIProvider(
        streaming=True,
        ai_responses=[
            AIResponse(
                content="Looking.",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="tc1", name="lookup", arguments={"q": "x"})],
            ),
            AIResponse(content="Found.", finish_reason="stop"),
        ],
    )

    async def handler(name: str, args: dict[str, Any]) -> str:
        return "ok"

    transport = SimpleChannel("sms1")
    kit.register_channel(transport)
    kit.register_channel(
        AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            tools=[AITool(name="lookup", description="Look up")],
        )
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    return transport


async def test_streamed_segment_applies_before_broadcast_modification() -> None:
    kit = RoomKit()
    await _wire(kit, content="Hi [PERSON_1], welcome")

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="deanon")
    async def deanon(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if not isinstance(event.content, TextContent) or "[PERSON_1]" not in event.content.body:
            return HookResult.allow()
        restored = event.content.body.replace("[PERSON_1]", "Alice")
        return HookResult.modify(event.model_copy(update={"content": TextContent(body=restored)}))

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )

    ai_msgs = _ai_messages(await kit.store.list_events("r1"))
    assert ai_msgs, "streamed AI segment was not persisted"
    assert ai_msgs[-1].content.body == "Hi Alice, welcome"
    assert "[PERSON_1]" not in ai_msgs[-1].content.body


async def test_streamed_segment_dropped_when_before_broadcast_blocks() -> None:
    kit = RoomKit()
    await _wire(kit, content="[SECRET] leaked")

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="blocker")
    async def blocker(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.source.channel_id == "ai1":
            return HookResult.block("withheld")
        return HookResult.allow()

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )

    # The blocked segment never enters the timeline as a delivered message.
    assert _ai_messages(await kit.store.list_events("r1")) == []


async def test_streaming_transport_persists_modified_but_streams_raw() -> None:
    """With a real streaming transport attached: the persisted segment is
    de-anonymised by BEFORE_BROADCAST (C1 works on the streaming path), while
    the live chunks already carried the raw text — they precede the hook by
    construction. The raw-chunk exposure is what a host-side gate prevents by
    withholding the stream fn when PII is active."""
    kit = RoomKit()
    provider = MockAIProvider(
        streaming=True,
        ai_responses=[AIResponse(content="Hi [PERSON_1]", finish_reason="stop")],
    )
    ws = WebSocketChannel("ws1")
    chunks: list[str] = []

    async def send_fn(conn_id: str, event: RoomEvent) -> None:
        return None

    async def stream_send_fn(conn_id: str, msg: object) -> None:
        delta = getattr(msg, "delta", None)
        if delta:
            chunks.append(delta)

    ws.register_connection("c1", send_fn, stream_send_fn=stream_send_fn, room_id="r1")
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(ws)
    kit.register_channel(AIChannel("ai1", provider=provider))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ws1")  # streaming-capable transport
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="deanon")
    async def deanon(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if isinstance(event.content, TextContent) and "[PERSON_1]" in event.content.body:
            restored = event.content.body.replace("[PERSON_1]", "Alice")
            return HookResult.modify(
                event.model_copy(update={"content": TextContent(body=restored)})
            )
        return HookResult.allow()

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )

    ai_msgs = _ai_messages(await kit.store.list_events("r1"))
    assert ai_msgs and ai_msgs[-1].content.body == "Hi Alice"  # persisted = de-anon
    assert "".join(chunks) == "Hi [PERSON_1]"  # live chunks = raw (by design)


@pytest.mark.parametrize("action", ["allow", "modify", "block"])
@pytest.mark.parametrize("live_transport", [False, True])
async def test_streamed_hook_side_effects_survive_decision(
    action: Literal["allow", "modify", "block"], live_transport: bool
) -> None:
    """Side effects run once, even for a blocked segment, on either delivery path."""
    kit = RoomKit()
    await _wire(kit, content="original")
    audit = SimpleChannel("audit")
    kit.register_channel(audit)
    await kit.attach_channel("r1", "audit")
    chunks: list[str] = []
    if live_transport:
        ws = WebSocketChannel("ws1")

        async def send_fn(conn_id: str, event: RoomEvent) -> None:
            pass

        async def stream_send_fn(conn_id: str, msg: object) -> None:
            delta = getattr(msg, "delta", None)
            if delta:
                chunks.append(delta)

        ws.register_connection("c1", send_fn, stream_send_fn=stream_send_fn, room_id="r1")
        kit.register_channel(ws)
        await kit.attach_channel("r1", "ws1")

    hook_calls: list[str] = []
    created_tasks: list[str] = []
    after_tasks: list[list[str]] = []
    task = Task(id="follow-up", room_id="r1", title="Follow up")
    observation = Observation(id="seen", room_id="r1", channel_id="ai1", content="Observed")
    injected = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
        content=TextContent(body="audit notice"),
    )

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def side_effects(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.source.channel_id != "ai1":
            return HookResult.allow()
        hook_calls.append(event.id)
        return HookResult(
            action=action,
            reason="withheld" if action == "block" else None,
            event=event.model_copy(update={"content": TextContent(body="modified")})
            if action == "modify"
            else None,
            tasks=[task],
            observations=[observation],
            injected_events=[InjectedEvent(event=injected, target_channel_ids=["audit"])],
        )

    @kit.hook(HookTrigger.ON_TASK_CREATED)
    async def task_created(event: RoomEvent, ctx: RoomContext) -> None:
        created_tasks.append(event.metadata["task_id"])

    @kit.hook(HookTrigger.AFTER_BROADCAST)
    async def after_broadcast(event: RoomEvent, ctx: RoomContext) -> None:
        if event.source.channel_id == "ai1":
            after_tasks.append([t.id for t in await kit.store.list_tasks("r1")])

    try:
        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
        assert len(hook_calls) == 1
        assert await kit.store.list_tasks("r1") == [task]
        assert await kit.store.list_observations("r1") == [observation]
        assert created_tasks == [task.id]
        events = await kit.store.list_events("r1")
        injections = [e for e in events if e.id == injected.id]
        assert len(injections) == 1
        assert [e.id for e in audit.delivered].count(injected.id) == 1
        replies = _ai_messages(events)
        if action == "block":
            assert replies == []
            assert after_tasks == []
        else:
            assert len(replies) == 1
            assert replies[0].content.body == ("modified" if action == "modify" else "original")
            assert replies[0].index < injections[0].index
            assert after_tasks == [[task.id]]
        if live_transport:
            assert "".join(chunks) == "original"
    finally:
        await kit.close()


@pytest.mark.parametrize("detached", [False, True])
async def test_stream_hook_effects_without_persistence_or_source_binding(detached: bool) -> None:
    """Effects survive when the segment has no stored row or no delivery plan."""
    kit = RoomKit(
        persistence_policy=None
        if detached
        else PersistencePolicy(exclude_types={EventType.MESSAGE})
    )
    audit = SimpleChannel("audit")
    kit.register_channel(audit)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "audit")
    if not detached:
        kit.register_channel(SimpleChannel("ai1", channel_type=ChannelType.AI))
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    task = Task(id="follow-up", room_id="r1", title="Follow up")
    observation = Observation(id="seen", room_id="r1", channel_id="ai1", content="Observed")
    notice = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
        content=TextContent(body="notice"),
    )

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def side_effects(event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult(
            action="allow",
            tasks=[task],
            observations=[observation],
            injected_events=[InjectedEvent(event=notice, target_channel_ids=["audit"])],
        )

    async def stream() -> AsyncIterator[str]:
        yield "answer"

    try:
        response = StreamingResponse(
            stream=stream(),
            source_channel_id="ai1",
            source_channel_type=ChannelType.AI,
            trigger_event=notice,
        )
        await kit._handle_streaming_response(
            kit._get_router(), response, "r1", await kit._build_context("r1")
        )
        assert await kit.store.list_tasks("r1") == [task]
        assert await kit.store.list_observations("r1") == [observation]
        events = await kit.store.list_events("r1")
        assert len(_ai_messages(events)) == int(detached)
        assert [event.id for event in events].count(notice.id) == 1
        assert [event.id for event in audit.delivered].count(notice.id) == 1
    finally:
        await kit.close()


async def test_stream_hook_tasks_wait_for_delivery_and_finish_before_return() -> None:
    """A slow transport keeps the turn pending through its post-delivery effects."""
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowTransport(SimpleChannel):
        async def deliver(
            self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
        ) -> ChannelOutput:
            if event.source.channel_id == "ai1":
                entered.set()
                await release.wait()
            return await super().deliver(event, binding, context)

    kit = RoomKit()
    await _wire(kit, content="answer")
    kit.register_channel(SlowTransport("slow"))
    await kit.attach_channel("r1", "slow")
    follow_up = Task(id="follow-up", room_id="r1", title="Follow up")

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def side_effects(event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult(
            action="allow", tasks=[follow_up] if event.source.channel_id == "ai1" else []
        )

    processing = asyncio.create_task(
        kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert not processing.done()
        assert await kit.store.list_tasks("r1") == []
        release.set()
        await asyncio.wait_for(processing, 2)
        assert await kit.store.list_tasks("r1") == [follow_up]
    finally:
        release.set()
        await asyncio.gather(processing, return_exceptions=True)
        await kit.close()


async def test_streamed_tool_call_events_carry_before_broadcast_modification() -> None:
    """The tool call's start and end cross the hooks like the text around them.

    Both used to commit straight from the stream: a hook that labels a tool
    call for the reader stamped the text segments and nothing else, so the
    stored rows and the non-streaming channels never saw the label.
    """
    kit = RoomKit()
    transport = await _wire_tool_turn(kit)

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="label")
    async def label(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.type not in _TOOL_EVENTS:
            return HookResult.allow()
        stamped = {**event.metadata, "label": f"Lookup · {event.type.value}"}
        return HookResult.modify(event.model_copy(update={"metadata": stamped}))

    try:
        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
        expected = ["Lookup · tool_call_start", "Lookup · tool_call_end"]
        stored = _tool_events(await kit.store.list_events("r1"))
        assert [e.type for e in stored] == list(_TOOL_EVENTS)
        assert [e.metadata["label"] for e in stored] == expected
        # The delivery carries the modified event too, not the raw one.
        assert [e.metadata.get("label") for e in _tool_events(transport.delivered)] == expected
    finally:
        await kit.close()


async def test_streamed_tool_call_events_dropped_when_before_broadcast_blocks() -> None:
    """A blocked tool-call event never lands, while its side effects do."""
    kit = RoomKit()
    transport = await _wire_tool_turn(kit)
    task = Task(id="follow-up", room_id="r1", title="Follow up")

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="blocker")
    async def blocker(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.type not in _TOOL_EVENTS:
            return HookResult.allow()
        # One task, handed over on the call's start alone.
        tasks = [task] if event.type == EventType.TOOL_CALL_START else []
        return HookResult(action="block", reason="withheld", tasks=tasks)

    try:
        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
        events = await kit.store.list_events("r1")
        assert _tool_events(events) == []
        assert _tool_events(transport.delivered) == []
        # The turn's text is untouched by a block aimed at its tool calls.
        assert [e.content.body for e in _ai_messages(events)] == ["Looking.", "Found."]
        assert await kit.store.list_tasks("r1") == [task]
    finally:
        await kit.close()
