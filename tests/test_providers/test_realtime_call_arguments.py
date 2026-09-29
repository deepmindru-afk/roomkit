"""Realtime providers read a call's arguments as every provider does (RFC §6.4).

A mapping, never an error: no arguments (nothing, JSON ``null``) are ``{}``,
anything that does not parse to an object is kept under ``raw``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState


@pytest.mark.parametrize(
    ("raw", "arguments"),
    [
        ('{"tz": "A"}', {"tz": "A"}),
        ("", {}),
        ("null", {}),
        ("[1, 2]", {"raw": "[1, 2]"}),
        ('{"tz": ', {"raw": '{"tz": '}),
    ],
)
async def test_openai_realtime_hands_a_mapping(raw: str, arguments: dict[str, Any]) -> None:
    provider = OpenAIRealtimeProvider(api_key="sk-test")
    session = VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="voice")
    provider._connections[session.id] = AsyncMock()
    provider._sessions[session.id] = session
    session.state = VoiceSessionState.ACTIVE
    seen: list[dict[str, Any]] = []
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

    assert seen == [arguments]
