"""The pipeline writes who a response comes from (RFC §8.5)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import pytest

from roomkit import RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.channels.base import Channel
from roomkit.core.event_router import unanswered
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventDirection,
    VerificationStatus,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


class _Impostor(Channel):
    """An agent whose answers claim to be an SMS sender's inbound message,
    in another room, at depth 0, under a key."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()
        answer = RoomEvent(
            room_id="elsewhere",
            source=EventSource(
                channel_id="sms",
                channel_type=ChannelType.SMS,
                direction=EventDirection.INBOUND,
                participant_id="customer",
                verification=VerificationStatus.VERIFIED,
            ),
            content=TextContent(body="answer"),
            idempotency_key="sms-42",
        )
        return ChannelOutput(responded=True, response_events=[answer])

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


async def _answered(caplog: pytest.LogCaptureFixture) -> tuple[RoomEvent, RoomEvent]:
    """The trigger and the answer the impostor gave it, as committed."""
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    for channel in (SimpleChannel("sms"), _Impostor("ai")):
        kit.register_channel(channel)
        await kit.attach_channel("r1", channel.channel_id)
    try:
        with caplog.at_level(logging.WARNING, logger="roomkit.event_router"):
            result = await kit.process_inbound(
                InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="hi"))
            )
        events = await kit.store.list_events("r1")
    finally:
        await kit.close()
    assert result.event is not None
    answers = [e for e in events if e.responds_to == result.event.id]
    assert len(answers) == 1
    return result.event, answers[0]


async def test_a_response_is_committed_as_its_responder_s(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, answer = await _answered(caplog)
    assert (answer.source.channel_id, answer.source.channel_type) == ("ai", ChannelType.AI)


async def test_a_response_keeps_nothing_of_the_source_it_claimed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, answer = await _answered(caplog)
    assert (answer.source.participant_id, answer.source.verification) == (None, None)


async def test_a_response_travels_outbound(caplog: pytest.LogCaptureFixture) -> None:
    _, answer = await _answered(caplog)
    assert answer.source.direction == EventDirection.OUTBOUND


async def test_a_response_takes_its_trigger_s_room_and_next_depth(
    caplog: pytest.LogCaptureFixture,
) -> None:
    trigger, answer = await _answered(caplog)
    assert (answer.room_id, answer.chain_depth) == ("r1", trigger.chain_depth + 1)


async def test_a_response_answers_its_trigger(caplog: pytest.LogCaptureFixture) -> None:
    trigger, answer = await _answered(caplog)
    assert answer.responds_to == trigger.id


class _AnswersTheFirst(Channel):
    """An agent whose every answer names the first event it was asked about."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.asked: list[str] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()
        self.asked.append(event.id)
        answer = RoomEvent(
            room_id=event.room_id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=TextContent(body=f"answer {len(self.asked)}"),
            responds_to=self.asked[0],
        )
        return ChannelOutput(responded=True, response_events=[answer])

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


async def test_a_response_keeps_the_event_its_channel_names_it_answers() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    agent = _AnswersTheFirst("ai")
    for channel in (SimpleChannel("sms"), agent):
        kit.register_channel(channel)
        await kit.attach_channel("r1", channel.channel_id)
    try:
        for body in ("first", "second"):
            await kit.process_inbound(
                InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body=body))
            )
        events = await kit.store.list_events("r1")
    finally:
        await kit.close()
    second = next(e for e in events if _body(e) == "answer 2")
    assert (second.responds_to, second.chain_depth) == (agent.asked[0], 1)


def _body(event: RoomEvent) -> str | None:
    return event.content.body if isinstance(event.content, TextContent) else None


async def test_a_response_carries_no_idempotency_key(caplog: pytest.LogCaptureFixture) -> None:
    _, answer = await _answered(caplog)
    assert answer.idempotency_key is None


async def test_a_response_claiming_another_source_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    await _answered(caplog)
    assert any("under another source" in r.getMessage() for r in caplog.records)


class _StreamTarget(SimpleChannel):
    """A transport that renders streamed answers and records the event it
    renders each one under."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.rendered_under: list[RoomEvent] = []

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[str],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        self.rendered_under.append(event)
        async for _ in text_stream:
            pass
        return ChannelOutput.empty()


async def _streamed() -> tuple[list[RoomEvent], _StreamTarget]:
    """What a streamed answer committed, and the target that rendered it."""
    kit = RoomKit()
    provider = MockAIProvider(
        streaming=True, ai_responses=[AIResponse(content="answer", finish_reason="stop")]
    )
    target = _StreamTarget("ws")
    await kit.create_room(room_id="r1")
    for channel in (SimpleChannel("sms"), target, AIChannel("ai", provider=provider)):
        kit.register_channel(channel)
    await kit.attach_channel("r1", "sms")
    await kit.attach_channel("r1", "ws")
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
    try:
        await kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="hi"))
        )
        events = await kit.store.list_events("r1")
    finally:
        await kit.close()
    return [e for e in events if e.source.channel_id == "ai"], target


async def test_a_streamed_answer_travels_outbound() -> None:
    answers, _ = await _streamed()
    assert [e.source.direction for e in answers] == [EventDirection.OUTBOUND]


async def test_a_streamed_answer_is_rendered_under_an_outbound_event() -> None:
    _, target = await _streamed()
    assert [e.source.direction for e in target.rendered_under] == [EventDirection.OUTBOUND]


def test_the_stand_in_for_an_unanswered_turn_travels_outbound() -> None:
    trigger = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
        content=TextContent(body="hi"),
    )
    stand_in = unanswered(trigger, "ai", ChannelType.AI)
    assert stand_in.source.direction == EventDirection.OUTBOUND
