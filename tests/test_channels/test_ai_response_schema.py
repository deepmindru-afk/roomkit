"""An AIChannel turn constrained to a response schema (RFC §6.7, RMK-249).

The schema resolves like the other per-turn settings: the binding metadata
wins, then the config provider, then the channel default. A turn whose provider
cannot honour it, or cannot honour it beside the turn's tools, fails before any
request is sent.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from roomkit import ResponseSchemaError
from roomkit.channels._turn_config import AIChannelTurnConfig
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AITool
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event

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
