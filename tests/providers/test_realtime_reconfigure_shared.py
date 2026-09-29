"""The in-band reconfigure shared by OpenAI and xAI (RMK-311 review).

A live reconfigure keeps only the provider config it applied: a key it cannot
change mid-session (turn detection, transcription) is named in a warning and
never recorded as if it were in effect.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.providers.anam.realtime import AnamRealtimeProvider
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.providers.personaplex.realtime import PersonaPlexRealtimeProvider
from roomkit.providers.xai.realtime import XAIRealtimeProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

_PROVIDERS = [
    pytest.param(
        lambda: OpenAIRealtimeProvider(api_key="sk", model="gpt-realtime-2.1"), id="openai"
    ),
    pytest.param(lambda: XAIRealtimeProvider(api_key="xai"), id="xai"),
]


def _live(provider: Any) -> VoiceSession:
    session = VoiceSession(
        id="s1",
        room_id="r1",
        participant_id="u1",
        channel_id="v1",
        state=VoiceSessionState.ACTIVE,
    )
    ws = AsyncMock()
    ws.__aiter__ = MagicMock(return_value=iter([]))
    provider._connections[session.id] = ws
    provider._sessions[session.id] = session
    return session


@pytest.mark.parametrize("make", _PROVIDERS)
async def test_a_key_the_live_session_cannot_take_is_not_recorded(
    make: Any, caplog: pytest.LogCaptureFixture
) -> None:
    provider = make()
    session = _live(provider)

    stale = {"threshold": 0.9, "create_response": False}
    await provider.reconfigure(session, provider_config=stale)

    assert provider._provider_configs.get(session.id, {}) == {}
    assert "threshold" in caplog.text and "create_response" in caplog.text


@pytest.mark.parametrize("make", _PROVIDERS)
async def test_without_a_live_connection_nothing_is_sent(make: Any) -> None:
    provider = make()
    session = VoiceSession(
        id="s2",
        room_id="r1",
        participant_id="u1",
        channel_id="v1",
        state=VoiceSessionState.ACTIVE,
    )

    await provider.reconfigure(session, system_prompt="x")

    assert session.id not in provider._connections


def test_a_provider_keeping_the_reconnecting_default_says_it_cannot_reconfigure() -> None:
    """A production provider that inherits the base's disconnect-and-reconnect
    ``reconfigure`` declares ``supports_mid_session_reconfigure = False``."""
    for cls in (AnamRealtimeProvider, ElevenLabsRealtimeProvider, PersonaPlexRealtimeProvider):
        assert cls.reconfigure is RealtimeVoiceProvider.reconfigure
        assert cls.supports_mid_session_reconfigure.fget(None) is False  # type: ignore[attr-defined]
