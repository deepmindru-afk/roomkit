"""Adapters report submission, unsupported input, and uncertain SDK failures."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from roomkit.providers.anam.config import AnamConfig
from roomkit.providers.anam.realtime import AnamRealtimeProvider
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.providers.personaplex.realtime import PersonaPlexRealtimeProvider
from roomkit.voice.base import VoiceSession


@pytest.fixture
def session() -> VoiceSession:
    return VoiceSession(id="s", room_id="r", channel_id="voice", participant_id="p")


@pytest.mark.parametrize("silent", [False, True])
async def test_elevenlabs_reports_only_completed_send(session, silent) -> None:
    provider = ElevenLabsRealtimeProvider(
        ElevenLabsRealtimeConfig(api_key="test", agent_id="agent")
    )
    missing = await provider.inject_text(session, "result", silent=silent)
    assert missing.status == "not_sent" and missing.retryable
    conversation = AsyncMock()
    provider._conversations[session.id] = conversation
    result = await provider.inject_text(session, "result", silent=silent)
    assert result.status == "sent"
    send = conversation.send_contextual_update if silent else conversation.send_user_message
    send.assert_awaited_once_with("result")
    send.side_effect = ConnectionError("confirmation lost")
    with pytest.raises(ConnectionError):
        await provider.inject_text(session, "another", silent=silent)


async def test_anam_does_not_report_swallowed_failure_as_sent(session) -> None:
    provider = AnamRealtimeProvider(AnamConfig(api_key="test", persona_id="persona"))
    missing = await provider.inject_text(session, "result")
    assert missing.status == "not_sent" and missing.retryable
    sdk = AsyncMock()
    provider._states[session.id] = SimpleNamespace(anam_session=sdk)
    result = await provider.inject_text(session, "result")
    assert result.status == "sent"
    sdk.send_message.assert_awaited_once_with("result")
    sdk.send_message.side_effect = ConnectionError("confirmation lost")
    failure = await provider.inject_text(session, "another")
    assert failure.status == "unknown" and not failure.retryable


async def test_personaplex_unsupported_input_cannot_succeed(session) -> None:
    result = await PersonaPlexRealtimeProvider().inject_text(session, "result")
    assert result.status == "not_sent"
    assert result.reason == "voice_injection_unsupported" and not result.retryable
