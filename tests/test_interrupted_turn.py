"""A turn cut short delivers nothing twice and nothing from an earlier turn (RFC §6.4; RMK-156).

The room already holds each round's text as its own message. A turn the
provider interrupts after a round is an error; delivered once its loop ends,
it ends on the marker alone, while a streamed one keeps what it streamed.
Both loops report it on ON_AI_RESPONSE with ``error``, then ON_ERROR fires,
and both record how the turn ended on its last message (RMK-289). A turn
cancelled between rounds adds no terminal text. The history the model was
given is context, never this turn's output.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.core.event_router import _solicits
from roomkit.core.mixins._child_execution import _persist_response_events
from roomkit.models.channel import ChannelOutput
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import (
    INTERRUPTION_MARKER_KEY,
    EventSource,
    RoomEvent,
    TextContent,
    is_interruption_marker,
)
from roomkit.models.steering import Cancel
from roomkit.orchestration.strategies.supervisor import _extract_output_text
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel

T = AITool(name="t", description="a tool", parameters={"type": "object", "properties": {}})
MARKER = "[Response interrupted]"
HISTORY = [
    AIResponse(content="Earlier answer."),
    AIResponse(content="Second answer."),
    AIResponse(
        content="Looking.",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="c1", name="t", arguments={})],
    ),
    AIResponse(content="never"),
]


class _FailingAt(MockAIProvider):
    """Scripted answers, and a provider error on the given generation."""

    def __init__(self, fail_at: int, answers: list[AIResponse], *, streaming: bool) -> None:
        super().__init__(ai_responses=answers, streaming=streaming)
        self._fail_at = fail_at
        self._generations = 0

    async def generate(self, context: AIContext) -> AIResponse:
        # The mock's stream draws its answers from generate() too: one count
        # per generation, whichever loop runs.
        self._generations += 1
        if self._generations == self._fail_at:
            raise ProviderError("upstream 500 req_abc123", retryable=False, status_code=500)
        return await super().generate(context)


async def _ok(name: str, arguments: dict[str, Any]) -> str:
    return "ok"


async def _room(
    provider: MockAIProvider, tool_handler: Any = _ok
) -> tuple[RoomKit, AIChannel, list[Any]]:
    ai = AIChannel(
        "ai1", provider=provider, tools=[T], tool_handler=tool_handler, tool_search=False
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    responses: list[Any] = []

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC, name="spy")
    async def spy(event: Any, ctx: Any) -> None:
        responses.append(event)

    return kit, ai, responses


async def _say(kit: RoomKit, *bodies: str) -> None:
    for body in bodies:
        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u", content=TextContent(body=body))
        )
        await asyncio.sleep(0.05)


async def _ai_messages(kit: RoomKit) -> list[str]:
    events = await kit.store.list_events("r1")
    return [
        e.content.body
        for e in events
        if e.type == EventType.MESSAGE
        and e.source.channel_id == "ai1"
        and isinstance(e.content, TextContent)
    ]


def _looking(content: str = "Looking.") -> AIResponse:
    return AIResponse(
        content=content,
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="c1", name="t", arguments={})],
    )


async def test_an_interrupted_turn_replays_nothing(streaming: bool) -> None:
    kit, _, _ = await _room(_FailingAt(4, HISTORY, streaming=streaming))

    await _say(kit, "first", "second", "third")

    messages = await _ai_messages(kit)
    assert messages[:2] == ["Earlier answer.", "Second answer."]
    assert all("answer." not in m for m in messages[2:])
    assert sum(m.count("Looking.") for m in messages) == 1
    await kit.close()


async def test_an_interrupted_turn_ends_on_the_marker_alone() -> None:
    """Delivered once its loop ends, the turn closes on the marker and reports
    its transcript, the segments once each (RFC §6.4)."""
    kit, _, responses = await _room(_FailingAt(4, HISTORY, streaming=False))

    await _say(kit, "first", "second", "third")

    assert await _ai_messages(kit) == ["Earlier answer.", "Second answer.", "Looking.", MARKER]
    assert responses[-1].response_content == f"Looking.\n\n{MARKER}"
    assert "req_abc123" not in responses[-1].response_content
    await kit.close()


async def test_an_interrupted_streamed_turn_keeps_what_it_streamed() -> None:
    """A streamed turn adds no marker: what it streamed stays, and its error
    surfaces (RFC §6.4)."""
    kit, _, _ = await _room(_FailingAt(4, HISTORY, streaming=True))
    errors: list[Any] = []

    @kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC, name="card")
    async def card(event: Any, ctx: Any) -> None:
        errors.append(event)

    await _say(kit, "first", "second", "third")

    assert await _ai_messages(kit) == ["Earlier answer.", "Second answer.", "Looking."]
    assert len(errors) == 1
    await kit.close()


async def test_an_interrupted_turn_is_reported_then_raised(streaming: bool) -> None:
    """RMK-289: both loops report the turn on ON_AI_RESPONSE with ``error``
    and what its rounds used, then ON_ERROR fires (RFC §6.4)."""
    used = {"input_tokens": 120, "output_tokens": 7}
    answers = [_looking().model_copy(update={"usage": used}), AIResponse(content="never")]
    kit, _, responses = await _room(_FailingAt(2, answers, streaming=streaming))
    seen: list[str] = []

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC, name="order_ai")
    async def order_ai(event: Any, ctx: Any) -> None:
        seen.append("ON_AI_RESPONSE")

    @kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC, name="order_error")
    async def order_error(event: Any, ctx: Any) -> None:
        seen.append("ON_ERROR")

    await _say(kit, "go")

    [report] = responses
    assert report.loop_end_reason == "error"
    assert report.usage["input_tokens"] == 120
    assert report.usage["output_tokens"] == 7
    assert seen == ["ON_AI_RESPONSE", "ON_ERROR"]
    events = await kit.store.list_events("r1")
    last = [e for e in events if e.type == EventType.MESSAGE and e.source.channel_id == "ai1"][-1]
    assert last.metadata["loop_end_reason"] == "error"
    assert last.metadata["ai_usage"]["input_tokens"] == 120
    await kit.close()


async def test_a_streamed_turn_records_its_end_on_its_final_text() -> None:
    """RMK-289: the documented read of ``loop_end_reason`` off the last
    message works on a streamed turn that completed."""
    answers = [
        _looking().model_copy(update={"usage": {"input_tokens": 10, "output_tokens": 2}}),
        AIResponse(content="Found it.", usage={"input_tokens": 30, "output_tokens": 4}),
    ]
    kit, _, _ = await _room(MockAIProvider(ai_responses=answers, streaming=True))

    await _say(kit, "go")

    events = await kit.store.list_events("r1")
    messages = [e for e in events if e.type == EventType.MESSAGE and e.source.channel_id == "ai1"]
    assert [m.content.body for m in messages] == ["Looking.", "Found it."]
    assert "loop_end_reason" not in messages[0].metadata
    assert messages[-1].metadata["loop_end_reason"] == "completed"
    assert messages[-1].metadata["ai_usage"] == {"input_tokens": 40, "output_tokens": 6}
    await kit.close()


async def test_an_interrupted_round_without_text_keeps_its_calls(streaming: bool) -> None:
    kit, _, _ = await _room(
        _FailingAt(2, [_looking(""), AIResponse(content="never")], streaming=streaming)
    )

    await _say(kit, "go")

    events = await kit.store.list_events("r1")
    kinds = [e.type for e in events if e.source.channel_id == "ai1"]
    assert EventType.TOOL_CALL_START in kinds
    assert EventType.TOOL_CALL_END in kinds
    # The marker closes a turn delivered once its loop ends (RFC §6.4).
    assert await _ai_messages(kit) == ([] if streaming else [MARKER])
    await kit.close()


async def test_a_turn_cancelled_between_rounds_adds_no_terminal_text(streaming: bool) -> None:
    ai: AIChannel | None = None

    async def cancelling(name: str, arguments: dict[str, Any]) -> str:
        assert ai is not None
        ai.steer(Cancel())
        return "ok"

    provider = MockAIProvider(
        ai_responses=[_looking(), AIResponse(content="never")], streaming=streaming
    )
    kit, ai, responses = await _room(provider, tool_handler=cancelling)

    await _say(kit, "go")

    assert await _ai_messages(kit) == ["Looking."]
    assert responses[-1].response_content == "Looking."
    await kit.close()


async def test_a_turn_without_final_text_keeps_its_record_on_its_last_message(
    streaming: bool,
) -> None:
    """The end reason and the usage ride the turn's last message when the
    turn has no final text to carry them (here, a cancel between rounds)."""
    ai: AIChannel | None = None

    async def cancelling(name: str, arguments: dict[str, Any]) -> str:
        assert ai is not None
        ai.steer(Cancel())
        return "ok"

    provider = MockAIProvider(
        ai_responses=[_looking(), AIResponse(content="never")], streaming=streaming
    )
    kit, ai, _ = await _room(provider, tool_handler=cancelling)

    await _say(kit, "go")

    events = await kit.store.list_events("r1")
    last = [e for e in events if e.type == EventType.MESSAGE and e.source.channel_id == "ai1"][-1]
    assert last.metadata["loop_end_reason"] == "cancelled"
    assert "ai_usage" in last.metadata
    await kit.close()


@pytest.mark.parametrize(
    "answers",
    [
        # The round's answer comes back empty, and its re-prompt fails.
        [_looking(), AIResponse(content="")],
        # Six identical calls pull the anti-loop ripcord; its final
        # generation fails.
        [_looking() for _ in range(6)],
    ],
    ids=["empty-answer-retry", "force-stop"],
)
async def test_every_generation_after_a_round_keeps_the_round(
    answers: list[AIResponse], streaming: bool
) -> None:
    fail_at = len(answers) + 1
    provider = _FailingAt(fail_at, answers, streaming=streaming)
    kit, _, _ = await _room(provider)

    await _say(kit, "go")

    events = await kit.store.list_events("r1")
    kinds = [e.type for e in events if e.source.channel_id == "ai1"]
    assert EventType.TOOL_CALL_END in kinds
    messages = await _ai_messages(kit)
    # The marker closes a turn delivered once its loop ends (RFC §6.4).
    if streaming:
        assert MARKER not in messages
    else:
        assert messages[-1] == MARKER
    await kit.close()


async def test_an_interrupted_turn_is_an_error(streaming: bool) -> None:
    """The turn is delivered and it is an error: ON_ERROR fires and the
    caller reads the provider's error, on both loops."""
    kit, _, _ = await _room(
        _FailingAt(2, [_looking(), AIResponse(content="never")], streaming=streaming)
    )
    errors: list[Any] = []

    @kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC, name="card")
    async def card(event: Any, ctx: Any) -> None:
        errors.append(event)

    result = await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u", content=TextContent(body="go"))
    )
    await asyncio.sleep(0.05)

    assert isinstance(result.error, ProviderError)
    assert len(errors) == 1
    assert "Looking." in await _ai_messages(kit)
    await kit.close()


class _Watcher(SimpleChannel):
    """A second agent in the room: records the messages it is asked to act on."""

    category = ChannelCategory.INTELLIGENCE
    channel_type = ChannelType.AI

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.asked: list[str] = []

    async def on_event(self, event: RoomEvent, binding: Any, context: Any) -> ChannelOutput:
        if event.type == EventType.MESSAGE and isinstance(event.content, TextContent):
            self.asked.append(event.content.body)
        return ChannelOutput.empty()


class TestTheMarkerIsNoAnswer:
    """RMK-289: the interruption marker is marked, solicits no agent, and is
    never read as an agent's answer (RFC §6.4, §19.3)."""

    async def test_it_is_marked(self) -> None:
        kit, _, _ = await _room(_FailingAt(4, HISTORY, streaming=False))

        await _say(kit, "first", "second", "third")

        events = await kit.store.list_events("r1")
        ai = [e for e in events if e.type == EventType.MESSAGE and e.source.channel_id == "ai1"]
        assert [is_interruption_marker(e) for e in ai] == [False, False, False, True]

    def test_it_solicits_nobody(self) -> None:
        marker = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ai1", channel_type=ChannelType.AI),
            content=TextContent(body=MARKER),
            addressed_to=["ai2"],
            metadata={INTERRUPTION_MARKER_KEY: True, "_always_process": ["supervisor"]},
        )
        for channel_id in ("ai2", "supervisor", "ai3"):
            assert _solicits(marker, channel_id, source_is_agent=True) is False

    async def test_another_agent_is_not_asked_to_answer_it(self) -> None:
        kit, _, _ = await _room(
            _FailingAt(2, [_looking(""), AIResponse(content="never")], streaming=False)
        )
        watcher = _Watcher("ai2")
        kit.register_channel(watcher)
        await kit.attach_channel("r1", "ai2", category=ChannelCategory.INTELLIGENCE)

        await kit.process_inbound(
            InboundMessage(
                channel_id="sms1",
                sender_id="u",
                content=TextContent(body="go"),
                addressed_to=["ai1"],
            )
        )
        await asyncio.sleep(0.1)

        assert await _ai_messages(kit) == [MARKER]
        assert watcher.asked == []
        await kit.close()

    def test_a_voice_barge_in_record_is_not_one(self) -> None:
        """RFC §12.3.13's ``interrupted`` marks a spoken reply cut by a barge-in:
        a different record, which is no interruption marker."""
        spoken = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="voice", channel_type=ChannelType.VOICE),
            content=TextContent(body="Your balance is"),
            metadata={"interrupted": True, "played_ms": 800},
        )

        assert is_interruption_marker(spoken) is False

    async def test_a_supervisor_does_not_hand_it_on_as_a_task(self) -> None:
        source = EventSource(channel_id="ai1", channel_type=ChannelType.AI)
        narration = RoomEvent(room_id="r1", source=source, content=TextContent(body="Looking."))
        marker = RoomEvent(
            room_id="r1",
            source=source,
            content=TextContent(body=MARKER),
            metadata={INTERRUPTION_MARKER_KEY: True},
        )

        assert await _extract_output_text(ChannelOutput(response_events=[marker])) == ""
        # An output that carries an error has no answer, its narration included
        interrupted = ChannelOutput(
            response_events=[narration, marker], error=ProviderError("upstream 500")
        )
        assert await _extract_output_text(interrupted) == ""


class TestADelegatedTurn:
    """RMK-289: a delegated turn reads the same whichever loop ran it."""

    async def test_an_interrupted_worker_fails_the_task(self, streaming: bool) -> None:
        answers = [_looking(), AIResponse(content="never")]
        worker = AIChannel(
            "worker",
            provider=_FailingAt(2, answers, streaming=streaming),
            tools=[T],
            tool_handler=_ok,
            tool_search=False,
        )
        kit = RoomKit()
        kit.register_channel(worker)
        await kit.create_room(room_id="parent")

        task = await kit.delegate("parent", "worker", "do the task", wait=True)

        assert task.result.status == "failed"
        assert "upstream 500" in (task.result.error or "")
        await kit.close()

    async def test_a_worker_returns_its_answer_and_records_its_end(self, streaming: bool) -> None:
        answers = [
            _looking().model_copy(update={"usage": {"input_tokens": 3, "output_tokens": 1}}),
            AIResponse(content="Done.", usage={"input_tokens": 5, "output_tokens": 2}),
        ]
        worker = AIChannel(
            "worker",
            provider=MockAIProvider(ai_responses=answers, streaming=streaming),
            tools=[T],
            tool_handler=_ok,
            tool_search=False,
        )
        kit = RoomKit()
        kit.register_channel(worker)
        await kit.create_room(room_id="parent")

        task = await kit.delegate("parent", "worker", "do the task", wait=True)

        assert task.result.output == "Done."
        events = await kit.store.list_events(task.child_room_id)
        rows = [
            e for e in events if e.type == EventType.MESSAGE and e.source.channel_id == "worker"
        ]
        assert rows[-1].metadata["loop_end_reason"] == "completed"
        assert rows[-1].metadata["ai_usage"] == {"input_tokens": 8, "output_tokens": 3}
        await kit.close()

    async def test_the_marker_alone_is_no_result(self) -> None:
        kit = RoomKit()
        await kit.create_room(room_id="child")
        marker = RoomEvent(
            room_id="child",
            source=EventSource(channel_id="ai1", channel_type=ChannelType.AI),
            content=TextContent(body=MARKER),
            metadata={INTERRUPTION_MARKER_KEY: True},
        )

        assert await _persist_response_events(kit, "child", [marker]) is None
        await kit.close()


class TestTheTurnRecord:
    """RMK-289: every turn records its end on its last message (RFC §6.4)."""

    async def test_a_turn_without_tools_records_it(self, streaming: bool) -> None:
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(content="Hi.", usage={"input_tokens": 5, "output_tokens": 1})
            ],
            streaming=streaming,
        )
        kit, _, _ = await _room(provider)

        result = await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u", content=TextContent(body="go"))
        )

        [reply] = [e for e in result.response_events if e.type == EventType.MESSAGE]
        assert reply.metadata["loop_end_reason"] == "completed"
        assert reply.metadata["ai_usage"] == {"input_tokens": 5, "output_tokens": 1}
        await kit.close()

    async def test_the_update_of_a_written_message_is_seen(self) -> None:
        """A streamed turn with no final text updates its last stored message:
        ON_EVENT_UPDATED sees it, as any change to a stored event."""
        ai: AIChannel | None = None

        async def cancelling(name: str, arguments: dict[str, Any]) -> str:
            assert ai is not None
            ai.steer(Cancel())
            return "ok"

        provider = MockAIProvider(
            ai_responses=[_looking(), AIResponse(content="never")], streaming=True
        )
        kit, ai, _ = await _room(provider, tool_handler=cancelling)
        updated: list[Any] = []

        @kit.hook(HookTrigger.ON_EVENT_UPDATED, execution=HookExecution.ASYNC, name="seen")
        async def seen(event: Any, ctx: Any) -> None:
            updated.append(event)

        await _say(kit, "go")

        [event] = updated
        assert event.content.body == "Looking."
        assert event.metadata["loop_end_reason"] == "cancelled"
        await kit.close()

    async def test_a_record_that_cannot_be_written_does_not_fail_the_turn(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        ai: AIChannel | None = None

        async def cancelling(name: str, arguments: dict[str, Any]) -> str:
            assert ai is not None
            ai.steer(Cancel())
            return "ok"

        async def broken(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("store down")

        provider = MockAIProvider(
            ai_responses=[_looking(), AIResponse(content="never")], streaming=True
        )
        kit, ai, responses = await _room(provider, tool_handler=cancelling)
        monkeypatch.setattr(kit, "update_event", broken)

        result = await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u", content=TextContent(body="go"))
        )

        assert result.error is None
        assert await _ai_messages(kit) == ["Looking."]
        assert "store down" in caplog.text
        await kit.close()
