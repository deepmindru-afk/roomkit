"""What every realtime provider owes its tool calls (RFC §12.4, RMK-299).

One scenario per provider, each on its own fake transport:

- a call the provider abandons (Gemini's cancellation, ElevenLabs' wait timing
  out, GPT-Live's restart) is reported to the channel once;
- a failed call's result travels as an error where the protocol can say so;
- the tasks a session lives on run in a context of their own;
- a provider whose model calls no tool is declared none;
- Gemini Live tells the model, once, that a call it could not parse did not run.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from roomkit import ConferenceRealtimeConfig, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.providers.ai.tool_calls import MALFORMED_CALL_NUDGE
from roomkit.providers.anam.config import AnamConfig
from roomkit.providers.anam.realtime import AnamRealtimeProvider
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.providers.openai.live_config import HostedReasoning
from roomkit.providers.personaplex.realtime import PersonaPlexRealtimeProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.test_openai_live import TOOL, _FakeWS, _function_call, _provider, _started
from tests.test_providers import test_gemini_realtime as gemini_tests
from tests.test_providers.test_gemini_realtime import (
    _blocking_call_state,
    _load_provider,
    _make_session,
)

Told = list[list[str]]
_ELEVENLABS = ElevenLabsRealtimeConfig(api_key="xi-test", agent_id="agent")


def _session() -> VoiceSession:
    return VoiceSession(
        id="s1",
        room_id="r1",
        participant_id="u1",
        channel_id="rt",
        state=VoiceSessionState.CONNECTING,
    )


# -- An abandoned call is reported to the channel, once -----------------------


async def _gemini_cancels() -> tuple[Told, list[str]]:
    provider, session, _state, _live = _blocking_call_state()
    told: Told = []
    provider.on_tool_call_cancelled(lambda s, ids: told.append(list(ids)))

    cancellation = SimpleNamespace(tool_call_cancellation=SimpleNamespace(ids=["call-1"]))
    await provider._handle_server_response(session, cancellation)

    return told, ["call-1"]


async def _elevenlabs_times_out() -> tuple[Told, list[str]]:
    provider = ElevenLabsRealtimeProvider(_ELEVENLABS.model_copy(update={"tool_timeout_s": 0.01}))
    session = _session()
    told: Told = []
    provider.on_tool_call_cancelled(lambda s, ids: told.append(list(ids)))
    handler = provider._make_tool_handler(session, "get_weather")

    with pytest.raises(RuntimeError, match="did not return"):
        await handler({"tool_call_id": "c1"})

    return told, ["c1"]


async def _gpt_live_restarts() -> tuple[Told, list[str]]:
    provider = _provider(delegation=HostedReasoning(model="gpt-5.6-terra"), close_timeout_s=0)
    session = _session()
    told: Told = []
    provider.on_tool_call_cancelled(lambda s, ids: told.append(list(ids)))
    first, second = _FakeWS(), _FakeWS()
    first.push(_started())
    second.push(_started())

    with patch("websockets.connect", AsyncMock(side_effect=[first, second])):
        await provider.connect(session, tools=[TOOL])
        first.push(_function_call("call_1", "get_weather", "{}"))
        await asyncio.sleep(0.01)
        await provider.reconfigure(session, voice="cedar")

    return told, ["call_1"]


_ABANDONS: dict[str, Callable[[], Awaitable[tuple[Told, list[str]]]]] = {
    "gemini-cancellation": _gemini_cancels,
    "elevenlabs-timeout": _elevenlabs_times_out,
    "gpt-live-restart": _gpt_live_restarts,
}


@pytest.mark.parametrize("scenario", list(_ABANDONS.values()), ids=list(_ABANDONS))
async def test_an_abandoned_call_is_reported_once(
    scenario: Callable[[], Awaitable[tuple[Told, list[str]]]],
) -> None:
    told, call_ids = await scenario()

    assert told == [call_ids]


# -- A failed call's result travels as an error where the protocol can say so -


async def test_elevenlabs_sends_a_failed_result_as_an_error() -> None:
    provider = ElevenLabsRealtimeProvider(_ELEVENLABS)
    session = _session()
    error = '{"error": "Tool \'get_weather\' is not permitted"}'
    provider.on_tool_call(
        lambda s, call_id, name, args: asyncio.ensure_future(
            provider.submit_tool_error(s, call_id, error)
        )
    )
    handler = provider._make_tool_handler(session, "get_weather")

    # The SDK sends a raising handler's text with ``is_error`` set.
    with pytest.raises(Exception) as raised:
        await handler({"tool_call_id": "c1"})

    assert str(raised.value) == error


async def test_a_protocol_without_errors_sends_the_failed_result_as_a_result() -> None:
    provider = MockRealtimeProvider()
    session = _session()

    await provider.submit_tool_error(session, "c1", '{"error": "no"}')

    assert provider.tool_results == [(session.id, "c1", '{"error": "no"}')]


class _ErrorAware(MockRealtimeProvider):
    """Records which of the two submissions the channel chose."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []

    async def submit_tool_error(self, session: VoiceSession, call_id: str, result: str) -> None:
        self.errors.append(call_id)
        await super().submit_tool_error(session, call_id, result)


async def test_the_channel_submits_a_failed_call_as_an_error() -> None:
    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        return "found"

    provider = _ErrorAware()
    tools = [{"name": "lookup", "description": "d", "parameters": {"type": "object"}}]
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=tools,
        tool_handler=lookup,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u1", "ws")

    await provider.simulate_tool_call(session, "ok", "lookup", {})
    await provider.simulate_tool_call(session, "refused", "undeclared", {})
    await until(lambda: len(provider.tool_results) == 2)

    assert provider.errors == ["refused"]
    await kit.close()


async def test_a_conference_submits_a_failed_call_as_an_error() -> None:
    async def handler(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        raise RuntimeError("backend down")

    provider = _ErrorAware()
    config = ConferenceRealtimeConfig(
        provider=provider, tools=[{"name": "x"}], tool_handler=handler
    )
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    session = await channel._realtime.ensure_session(ROOM)
    assert session is not None

    await provider.simulate_tool_call(session, "call-1", "x", {})
    await until(lambda: bool(provider.tool_results))

    assert provider.errors == ["call-1"]
    await kit.close()


# -- The tasks a session lives on run in a context of their own ---------------

_CALLER: contextvars.ContextVar[str | None] = contextvars.ContextVar("caller", default=None)


async def test_a_gpt_live_receive_loop_does_not_inherit_its_starters_context() -> None:
    provider = _provider()
    session = _session()
    ws = _FakeWS()
    ws.push(_started())
    _CALLER.set("the handler's call")

    with patch("websockets.connect", AsyncMock(return_value=ws)):
        await provider.connect(session)

    task = provider._states[session.id].receive_task
    assert task is not None
    assert task.get_context().get(_CALLER) is None
    await provider.disconnect(session)


async def test_a_session_task_runs_in_a_context_of_its_own() -> None:
    seen: list[str | None] = []

    async def loop() -> None:
        seen.append(_CALLER.get())

    _CALLER.set("the handler's call")
    await MockRealtimeProvider._session_task(loop(), name="t")

    assert seen == [None]


# -- A provider whose model calls no tool is declared none --------------------


@pytest.mark.parametrize(
    ("build", "supports"),
    [
        (lambda: AnamRealtimeProvider(AnamConfig(api_key="k", persona_id="p")), False),
        (PersonaPlexRealtimeProvider, False),
        (lambda: ElevenLabsRealtimeProvider(_ELEVENLABS), True),
        (MockRealtimeProvider, True),
    ],
    ids=["anam", "personaplex", "elevenlabs", "mock"],
)
def test_supports_tools_says_whether_the_model_calls_tools(
    build: Callable[[], Any], supports: bool
) -> None:
    assert build().supports_tools is supports


class _Toolless(MockRealtimeProvider):
    @property
    def supports_tools(self) -> bool:
        return False


async def test_a_toolless_provider_is_declared_no_tool(caplog: pytest.LogCaptureFixture) -> None:
    provider = _Toolless()
    tools = [{"name": "lookup", "description": "d", "parameters": {"type": "object"}}]
    with caplog.at_level(logging.WARNING, logger="roomkit.channels.tools"):
        channel = RealtimeVoiceChannel(
            "rt", provider=provider, transport=MockRealtimeTransport(), tools=tools
        )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")

    await channel.start_session("r1", "u1", "ws")

    connect = next(c for c in provider.calls if c.method == "connect")
    assert not connect.args.get("tools")
    assert any("cannot call tools" in r.getMessage() for r in caplog.records)
    await kit.close()


# -- Gemini Live tells the model a call it could not parse did not run --------


async def test_gemini_tells_the_model_once_until_the_user_speaks_again() -> None:
    mod = _load_provider()
    provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
    session = _make_session()
    state = mod._GeminiSessionState(session=session)
    provider._sessions[session.id] = state
    provider.inject_text = AsyncMock()  # type: ignore[method-assign]
    malformed = gemini_tests.TestGeminiLiveProvider._content(
        turn_complete=True, turn_complete_reason=SimpleNamespace(name="MALFORMED_FUNCTION_CALL")
    )

    await provider._handle_server_response(session, malformed)
    await provider._handle_server_response(session, malformed)
    assert provider.inject_text.await_count == 1
    assert provider.inject_text.await_args.args[1] == MALFORMED_CALL_NUDGE

    await provider._on_voice_activity(
        session, state, SimpleNamespace(voice_activity_type="ACTIVITY_START")
    )
    await provider._handle_server_response(session, malformed)
    assert provider.inject_text.await_count == 2
