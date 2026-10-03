"""Realtime providers read a call's arguments as every provider does (RFC §6.4,
§12.4): with the shared rule, ``readable_arguments``.

No arguments (nothing, JSON ``null``) are ``{}`` and an object is that mapping;
anything that does not parse to an object (invalid JSON, an array, a fragment)
reaches ``on_tool_call`` as the text the model wrote, which the channel refuses.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from roomkit.providers.deepgram.config import DeepgramAgentConfig
from roomkit.providers.deepgram.realtime import DeepgramAgentProvider
from roomkit.providers.openai.live_config import HostedReasoning
from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState
from tests.conference.test_conference_realtime import until
from tests.test_openai_live import TOOL, _function_call, _provider, _response_event
from tests.test_openai_live import _connect as live_connect
from tests.test_realtime_deepgram import _connect as deepgram_connect

Reads = Callable[[str], Awaitable[list[Any]]]


def _session() -> VoiceSession:
    return VoiceSession(
        id="s1",
        room_id="r1",
        participant_id="p1",
        channel_id="voice",
        state=VoiceSessionState.CONNECTING,
    )


async def _openai_realtime_reads(raw: str) -> list[Any]:
    """OpenAI Realtime, and xAI, which shares its event handling."""
    provider = OpenAIRealtimeProvider(api_key="sk-test")
    session = _session()
    provider._connections[session.id] = AsyncMock()
    provider._sessions[session.id] = session
    session.state = VoiceSessionState.ACTIVE
    seen: list[Any] = []
    provider.on_tool_call(lambda s, call_id, name, args: seen.append(args))
    await provider._handle_server_event(
        session,
        {
            "type": "response.function_call_arguments.done",
            "call_id": "c1",
            "name": "now",
            "arguments": raw,
        },
    )
    return seen


async def _gpt_live_reads(raw: str) -> list[Any]:
    provider = _provider(delegation=HostedReasoning(model="gpt-5.6-terra"))
    session = _session()
    seen: list[Any] = []
    provider.on_tool_call(lambda s, call_id, name, args: seen.append(args))
    ws, _ = await live_connect(provider, session, tools=[TOOL])
    ws.push(_response_event({"type": "response.created"}))
    ws.push(_function_call("c1", "get_weather", raw))
    await until(lambda: bool(seen))
    await provider.disconnect(session)
    return seen


async def _deepgram_reads(raw: str) -> list[Any]:
    provider = DeepgramAgentProvider(DeepgramAgentConfig(api_key=SecretStr("dg-key")))
    session = _session()
    seen: list[Any] = []
    provider.on_tool_call(lambda s, call_id, name, args: seen.append(args))
    ws = await deepgram_connect(provider, session)
    call = {"id": "fc_1", "name": "now", "arguments": raw, "client_side": True}
    ws.push(json.dumps({"type": "FunctionCallRequest", "functions": [call]}))
    await until(lambda: bool(seen))
    await provider.disconnect(session)
    return seen


@pytest.mark.parametrize(
    "reads",
    [_openai_realtime_reads, _gpt_live_reads, _deepgram_reads],
    ids=["openai-realtime-and-xai", "gpt-live", "deepgram"],
)
@pytest.mark.parametrize(
    ("raw", "arguments"),
    [
        ('{"tz": "A"}', {"tz": "A"}),
        ("", {}),
        ("null", {}),
        ("[1, 2]", "[1, 2]"),
        ('{"tz": ', '{"tz": '),
    ],
)
async def test_a_realtime_provider_hands_a_mapping_or_the_unreadable_text(
    reads: Reads, raw: str, arguments: dict[str, Any] | str
) -> None:
    assert await reads(raw) == [arguments]
