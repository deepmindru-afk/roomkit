"""Which realtime providers reconfigure a live session in place (RMK-311).

A provider whose protocol only takes its settings when the connection opens
says so, and the channel then never reconfigures it mid-conversation for
Tool Search or a skill (a reconnect would start a new conversation).
"""

from __future__ import annotations

from roomkit.providers.anam.config import AnamConfig
from roomkit.providers.anam.realtime import AnamRealtimeProvider
from roomkit.providers.personaplex.realtime import PersonaPlexRealtimeProvider


def test_personaplex_does_not_reconfigure_mid_session() -> None:
    assert PersonaPlexRealtimeProvider().supports_mid_session_reconfigure is False


def test_anam_does_not_reconfigure_mid_session() -> None:
    provider = AnamRealtimeProvider(AnamConfig(api_key="test", persona_id="persona"))

    assert provider.supports_mid_session_reconfigure is False
