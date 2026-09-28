"""An AIChannel turn constrained to a response schema (RFC §6.7, RMK-249).

The schema resolves like the other per-turn settings: the binding metadata
wins, then the config provider, then the channel default. A turn whose provider
cannot honour it, or cannot honour it beside the turn's tools, fails before any
request is sent.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError

from roomkit import ResponseSchemaError
from roomkit.channels._turn_config import AIChannelTurnConfig
from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.core.hooks import HookRegistration
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.test_framework import SimpleChannel

TRIAGE: dict[str, Any] = {
    "type": "object",
    "properties": {"department": {"type": "string", "enum": ["billing", "other"]}},
    "required": ["department"],
    "additionalProperties": False,
}
OTHER: dict[str, Any] = {
    "type": "object",
    "properties": {"urgent": {"type": "boolean"}},
    "required": ["urgent"],
    "additionalProperties": False,
}
ANSWER = json.dumps({"department": "billing"})


def _binding(metadata: dict[str, Any] | None = None) -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata=metadata or {},
    )


def _ctx(binding: ChannelBinding) -> RoomContext:
    return RoomContext(room=Room(id="r1"), bindings=[binding])


class TestResolution:
    async def test_the_channel_default_reaches_the_context(self) -> None:
        channel = AIChannel("ai1", provider=MockAIProvider([ANSWER]), response_schema=TRIAGE)
        binding = _binding()

        context = await channel._build_context(make_event(), binding, _ctx(binding))

        assert context.response_schema == TRIAGE

    async def test_the_config_provider_overrides_the_default(self) -> None:
        async def provider(_binding: ChannelBinding, _context: RoomContext) -> Any:
            return AIChannelTurnConfig(response_schema=OTHER)

        channel = AIChannel(
            "ai1",
            provider=MockAIProvider([ANSWER]),
            response_schema=TRIAGE,
            config_provider=provider,
        )
        binding = _binding()

        context = await channel._build_context(make_event(), binding, _ctx(binding))

        assert context.response_schema == OTHER

    async def test_the_binding_metadata_wins(self) -> None:
        async def provider(_binding: ChannelBinding, _context: RoomContext) -> Any:
            return AIChannelTurnConfig(response_schema=OTHER)

        channel = AIChannel("ai1", provider=MockAIProvider([ANSWER]), config_provider=provider)
        binding = _binding({"response_schema": TRIAGE})

        context = await channel._build_context(make_event(), binding, _ctx(binding))

        assert context.response_schema == TRIAGE

    async def test_no_schema_anywhere_leaves_the_turn_free(self) -> None:
        channel = AIChannel("ai1", provider=MockAIProvider(["hi"]))
        binding = _binding()

        context = await channel._build_context(make_event(), binding, _ctx(binding))

        assert context.response_schema is None

    def test_a_default_outside_the_portable_subset_fails_at_construction(self) -> None:
        with pytest.raises(ValueError, match="additionalProperties"):
            AIChannel(
                "ai1",
                provider=MockAIProvider([ANSWER]),
                response_schema={"type": "object", "properties": {}, "required": []},
            )

    async def test_a_binding_schema_outside_the_subset_fails_the_turn(self) -> None:
        channel = AIChannel("ai1", provider=MockAIProvider([ANSWER], response_schema=True))
        binding = _binding({"response_schema": {"type": "string"}})

        with pytest.raises(ValidationError, match="root type"):
            await channel._build_context(make_event(), binding, _ctx(binding))


class TestTurn:
    async def test_the_answer_is_the_json_document(self) -> None:
        provider = MockAIProvider([ANSWER], response_schema=True)
        channel = AIChannel("ai1", provider=provider, response_schema=TRIAGE)
        binding = _binding()

        output = await channel.on_event(make_event(body="charged twice"), binding, _ctx(binding))

        assert json.loads(output.response_events[0].content.body) == {"department": "billing"}
        assert provider.calls[0].response_schema == TRIAGE

    async def test_a_provider_without_support_fails_the_turn_before_any_request(self) -> None:
        provider = MockAIProvider([ANSWER])
        channel = AIChannel("ai1", provider=provider, response_schema=TRIAGE)
        binding = _binding()

        with pytest.raises(ResponseSchemaError) as exc:
            await channel.on_event(make_event(body="charged twice"), binding, _ctx(binding))

        assert exc.value.reason == "unsupported"
        assert provider.calls == []

    async def test_tools_beside_the_schema_fail_the_turn_where_they_cannot_combine(
        self,
    ) -> None:
        provider = MockAIProvider([ANSWER], response_schema=True)

        async def handler(_name: str, _arguments: dict[str, Any]) -> str:
            return "{}"

        channel = AIChannel(
            "ai1",
            provider=provider,
            response_schema=TRIAGE,
            tools=[AITool(name="lookup", description="Look it up")],
            tool_handler=handler,
        )
        binding = _binding()

        with pytest.raises(ResponseSchemaError) as exc:
            await channel.on_event(make_event(body="charged twice"), binding, _ctx(binding))

        assert exc.value.reason == "unsupported"
        assert provider.calls == []

    async def test_tools_beside_the_schema_run_where_they_combine(self) -> None:
        provider = MockAIProvider([ANSWER], response_schema=True, response_schema_with_tools=True)

        async def handler(_name: str, _arguments: dict[str, Any]) -> str:
            return "{}"

        channel = AIChannel(
            "ai1",
            provider=provider,
            response_schema=TRIAGE,
            tools=[AITool(name="lookup", description="Look it up")],
            tool_handler=handler,
        )
        binding = _binding()

        output = await channel.on_event(make_event(body="charged twice"), binding, _ctx(binding))

        assert json.loads(output.response_events[0].content.body) == {"department": "billing"}


async def _turn_through_the_room(ai: AIChannel) -> tuple[Any, list[RoomEvent], SimpleChannel]:
    """One inbound turn in a room with an SMS channel and *ai*: the result, the
    ON_ERROR events, and the SMS channel that records what it was sent."""
    kit = RoomKit()
    sms = SimpleChannel("sms1")
    kit.register_channel(sms)
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    errors: list[RoomEvent] = []

    async def on_error(event: RoomEvent, _ctx: RoomContext) -> None:
        errors.append(event)

    kit.hook_engine.register(
        HookRegistration(
            trigger=HookTrigger.ON_ERROR,
            execution=HookExecution.ASYNC,
            fn=on_error,
            name="capture_error",
        )
    )
    result = await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )
    await asyncio.sleep(0.05)
    stored = await kit.store.list_events("r1")
    sms.stored_ai = [e for e in stored if e.source.channel_id == "ai1"]  # type: ignore[attr-defined]
    await kit.close()
    return result, errors, sms


class TestStreamedTurnThroughTheRoom:
    """A streamed constrained answer reaches the room only once it is checked."""

    async def test_an_answer_that_fails_the_check_is_never_stored_nor_sent(self) -> None:
        provider = MockAIProvider(
            ["Sure! The department is billing."], response_schema=True, streaming=True
        )

        result, errors, sms = await _turn_through_the_room(
            AIChannel("ai1", provider=provider, response_schema=TRIAGE)
        )

        assert isinstance(result.error, ResponseSchemaError)
        assert result.error.reason == "invalid_json"
        assert len(errors) == 1
        assert sms.delivered == []
        assert sms.stored_ai == []  # type: ignore[attr-defined]

    async def test_an_answer_that_passes_is_stored_and_sent_whole(self) -> None:
        provider = MockAIProvider([ANSWER], response_schema=True, streaming=True)

        result, errors, sms = await _turn_through_the_room(
            AIChannel("ai1", provider=provider, response_schema=TRIAGE)
        )

        assert result.error is None and errors == []
        assert [json.loads(e.content.body) for e in sms.delivered] == [{"department": "billing"}]


class _AlwaysCallsATool(MockAIProvider):
    """A model that never stops calling its tool."""

    def _next_response(self) -> AIResponse:
        return AIResponse(
            content="Let me look that up.",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})],
        )


class TestToolLoopCutShort:
    """A constrained turn whose tool loop stops before a final answer fails."""

    @staticmethod
    def _channel(streaming: bool) -> AIChannel:
        async def handler(_name: str, _arguments: dict[str, Any]) -> str:
            return "{}"

        return AIChannel(
            "ai1",
            provider=_AlwaysCallsATool(
                response_schema=True, response_schema_with_tools=True, streaming=streaming
            ),
            response_schema=TRIAGE,
            tools=[AITool(name="lookup", description="Look it up")],
            tool_handler=handler,
            max_tool_rounds=2,
        )

    async def test_the_non_streaming_turn_raises_truncated(self) -> None:
        binding = _binding()

        with pytest.raises(ResponseSchemaError) as exc:
            await self._channel(streaming=False).on_event(
                make_event(body="go"), binding, _ctx(binding)
            )

        assert exc.value.reason == "truncated"
        assert "max_rounds" in str(exc.value)

    async def test_the_streaming_turn_raises_truncated(self) -> None:
        binding = _binding()
        output = await self._channel(streaming=True).on_event(
            make_event(body="go"), binding, _ctx(binding)
        )

        with pytest.raises(ResponseSchemaError) as exc:
            async for _ in output.response_stream:
                pass

        assert exc.value.reason == "truncated"


class TestToolRoundNarration:
    """A tool round's own text reaches the room as its own message; the turn's
    last text message is the document."""

    @pytest.mark.parametrize("streaming", [False, True])
    async def test_the_narration_and_the_document_are_separate_messages(
        self, streaming: bool
    ) -> None:
        async def handler(_name: str, _arguments: dict[str, Any]) -> str:
            return "{}"

        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="Let me look that up.",
                    finish_reason="tool_calls",
                    tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})],
                ),
                AIResponse(content=ANSWER, finish_reason="stop"),
            ],
            response_schema=True,
            response_schema_with_tools=True,
            streaming=streaming,
        )
        ai = AIChannel(
            "ai1",
            provider=provider,
            response_schema=TRIAGE,
            tools=[AITool(name="lookup", description="Look it up")],
            tool_handler=handler,
        )

        result, _errors, sms = await _turn_through_the_room(ai)

        texts = [e.content.body for e in sms.delivered if isinstance(e.content, TextContent)]
        assert result.error is None
        assert texts == ["Let me look that up.", ANSWER]
