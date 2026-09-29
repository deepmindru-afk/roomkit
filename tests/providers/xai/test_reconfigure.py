"""XAIRealtimeProvider.reconfigure: in-band ``session.update``, no teardown (RMK-311).

xAI speaks the OpenAI Realtime protocol with a flat session config. Tool
Search (``find_tools``), skill activation and a handoff call ``reconfigure``
mid-conversation; a reconnect would throw the conversation away, so the
update travels in band, as on OpenAI.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.providers.xai.realtime import XAIRealtimeProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState


@pytest.fixture
def provider() -> XAIRealtimeProvider:
    return XAIRealtimeProvider(api_key="xai-test")


@pytest.fixture
def session() -> VoiceSession:
    return VoiceSession(
        id="s1",
        room_id="r1",
        participant_id="u1",
        channel_id="v1",
        state=VoiceSessionState.CONNECTING,
    )


def _live(provider: XAIRealtimeProvider, session: VoiceSession) -> AsyncMock:
    ws = AsyncMock()
    ws.__aiter__ = MagicMock(return_value=iter([]))
    provider._connections[session.id] = ws
    provider._sessions[session.id] = session
    session.state = VoiceSessionState.ACTIVE
    provider.disconnect = AsyncMock()  # type: ignore[method-assign]
    provider.connect = AsyncMock()  # type: ignore[method-assign]
    return ws


def _updates(ws: AsyncMock) -> list[dict]:
    return [json.loads(call.args[0]) for call in ws.send.call_args_list]


TOOLS = [{"type": "function", "name": "calendar", "description": "d", "parameters": {}}]


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"tools": TOOLS}, id="find_tools"),
        pytest.param({"system_prompt": "Rules.", "tools": TOOLS}, id="skill activation"),
        pytest.param({"system_prompt": "You are B.", "voice": "Rex", "tools": []}, id="handoff"),
    ],
)
async def test_a_reconfigure_keeps_the_conversation(
    provider: XAIRealtimeProvider, session: VoiceSession, change: dict
) -> None:
    ws = _live(provider, session)

    await provider.reconfigure(session, **change)

    provider.disconnect.assert_not_called()
    provider.connect.assert_not_called()
    (update,) = _updates(ws)
    assert update["type"] == "session.update"
    # xAI's flat shape: only the fields that change, none nested.
    expected = {"system_prompt": "instructions", "voice": "voice", "tools": "tools"}
    assert set(update["session"]) == {expected[key] for key in change}
    if "tools" in change:
        assert [t["name"] for t in update["session"]["tools"]] == [
            t["name"] for t in change["tools"]
        ]


async def test_a_reconfigure_that_changes_nothing_sends_nothing(
    provider: XAIRealtimeProvider, session: VoiceSession
) -> None:
    ws = _live(provider, session)

    await provider.reconfigure(session)

    ws.send.assert_not_called()


def test_xai_reconfigures_mid_session() -> None:
    """The channel reconfigures it in band (Tool Search, skills) rather than
    falling back to ``call_tool`` or inline skills."""
    assert XAIRealtimeProvider(api_key="xai-test").supports_mid_session_reconfigure
