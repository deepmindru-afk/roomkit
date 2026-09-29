"""Tool results on the OpenAI Realtime wire: one continuation per response (RMK-279).

The model is asked to go on (``response.create``) once per response, when that
response is done and every call it emitted has its output (RFC §12.4). xAI
speaks the same wire and inherits the behaviour.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.providers.openai.realtime_base import OpenAIRealtimeBase
from roomkit.providers.xai.config import XAIRealtimeConfig
from roomkit.providers.xai.realtime import XAIRealtimeProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState

_PROVIDERS: dict[str, Callable[[], OpenAIRealtimeBase]] = {
    "openai": lambda: OpenAIRealtimeProvider(api_key="sk-test"),
    "xai": lambda: XAIRealtimeProvider(XAIRealtimeConfig(api_key=SecretStr("xai-test"))),
}


@pytest.fixture(params=sorted(_PROVIDERS))
def provider(request: pytest.FixtureRequest) -> OpenAIRealtimeBase:
    return _PROVIDERS[request.param]()


@pytest.fixture
def session() -> VoiceSession:
    return VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="voice")


def _attach(provider: OpenAIRealtimeBase, session: VoiceSession) -> AsyncMock:
    ws = AsyncMock()
    provider._connections[session.id] = ws
    provider._sessions[session.id] = session
    session.state = VoiceSessionState.ACTIVE
    return ws


def _wire(ws: AsyncMock) -> list[str]:
    """What went out, as ``item(<call_id>)`` or the bare event type."""
    sent = []
    for call in ws.send.call_args_list:
        event = json.loads(call.args[0])
        if event["type"] == "conversation.item.create":
            sent.append(f"item({event['item']['call_id']})")
        else:
            sent.append(event["type"])
    return sent


async def _response_created(provider: OpenAIRealtimeBase, session: VoiceSession) -> None:
    await provider._handle_server_event(session, {"type": "response.created", "response": {}})


async def _call(provider: OpenAIRealtimeBase, session: VoiceSession, call_id: str) -> None:
    await provider._handle_server_event(
        session,
        {
            "type": "response.function_call_arguments.done",
            "call_id": call_id,
            "name": "lookup",
            "arguments": "{}",
        },
    )


async def _response_done(
    provider: OpenAIRealtimeBase, session: VoiceSession, status: str = "completed"
) -> None:
    await provider._handle_server_event(
        session, {"type": "response.done", "response": {"status": status}}
    )


async def _result(provider: OpenAIRealtimeBase, session: VoiceSession, call_id: str) -> None:
    await provider.submit_tool_result(session, call_id, '{"ok": true}')


class TestOneContinuationPerResponse:
    async def test_parallel_calls_answered_before_the_response_ends(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _call(provider, session, "call_b")
        await _result(provider, session, "call_a")
        await _result(provider, session, "call_b")
        assert _wire(ws) == ["item(call_a)", "item(call_b)"]

        await _response_done(provider, session)

        assert _wire(ws) == ["item(call_a)", "item(call_b)", "response.create"]

    async def test_parallel_calls_answered_after_the_response_ends(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _call(provider, session, "call_b")
        await _response_done(provider, session)
        await _result(provider, session, "call_b")
        assert _wire(ws) == ["item(call_b)"]

        await _result(provider, session, "call_a")

        assert _wire(ws) == ["item(call_b)", "item(call_a)", "response.create"]

    async def test_a_single_call_continues_after_its_result(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)

        await _result(provider, session, "call_a")

        assert _wire(ws) == ["item(call_a)", "response.create"]

    @pytest.mark.parametrize("status", ["completed", "cancelled", "failed"])
    async def test_a_response_ends_the_same_way_whatever_its_status(
        self, provider: OpenAIRealtimeBase, session: VoiceSession, status: str
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _call(provider, session, "call_b")
        await _result(provider, session, "call_a")
        await _response_done(provider, session, status)
        assert _wire(ws) == ["item(call_a)"]

        await _result(provider, session, "call_b")

        assert _wire(ws) == ["item(call_a)", "item(call_b)", "response.create"]

    async def test_a_result_submitted_inside_the_tool_call_callback(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)

        async def serve_at_once(sess: VoiceSession, call_id: str, name: str, args: dict) -> None:
            await provider.submit_tool_result(sess, call_id, '{"ok": true}')

        provider.on_tool_call(serve_at_once)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        assert _wire(ws) == ["item(call_a)"]

        await _response_done(provider, session)

        assert _wire(ws) == ["item(call_a)", "response.create"]

    async def test_a_response_without_calls_asks_nothing(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _response_done(provider, session)

        assert _wire(ws) == []

    async def test_the_continuation_opens_a_fresh_count(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _result(provider, session, "call_a")
        assert _wire(ws) == ["item(call_a)"]
        await _response_done(provider, session)
        # The continuation calls a tool of its own
        await _response_created(provider, session)
        await _call(provider, session, "call_b")
        await _response_done(provider, session)
        await _result(provider, session, "call_b")

        assert _wire(ws) == ["item(call_a)", "response.create", "item(call_b)", "response.create"]


class TestAResultTheConversationHasLeft:
    async def test_it_waits_for_the_response_in_progress(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)
        # The user speaks again before call_a is answered
        await _response_created(provider, session)
        await _result(provider, session, "call_a")
        assert _wire(ws) == ["item(call_a)"]

        await _response_done(provider, session)

        assert _wire(ws) == ["item(call_a)", "response.create"]

    async def test_it_does_not_hold_the_new_response(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _call(provider, session, "call_b")
        await _result(provider, session, "call_a")
        await _response_done(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_c")
        await _response_done(provider, session)

        await _result(provider, session, "call_c")

        assert _wire(ws) == ["item(call_a)", "item(call_c)", "response.create"]

    async def test_it_continues_at_once_when_nothing_is_in_progress(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _call(provider, session, "call_b")
        await _result(provider, session, "call_a")
        await _response_done(provider, session)
        await _response_created(provider, session)
        await _response_done(provider, session)

        await _result(provider, session, "call_b")

        assert _wire(ws) == ["item(call_a)", "item(call_b)", "response.create"]

    async def test_it_joins_the_new_response_calls(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_c")
        await _response_done(provider, session)
        await _result(provider, session, "call_a")
        assert _wire(ws) == ["item(call_a)"]

        await _result(provider, session, "call_c")

        assert _wire(ws) == ["item(call_a)", "item(call_c)", "response.create"]


class TestARequestNotYetBegun:
    async def test_a_result_in_between_waits_for_the_requested_response(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _call(provider, session, "call_b")
        await _response_done(provider, session)
        await _response_created(provider, session)  # the user spoke again
        await _response_done(provider, session)
        await _result(provider, session, "call_a")
        assert _wire(ws) == ["item(call_a)", "response.create"]

        # call_b's output lands before the requested response has begun
        await _result(provider, session, "call_b")
        assert _wire(ws) == ["item(call_a)", "response.create", "item(call_b)"]
        await _response_created(provider, session)
        assert _wire(ws) == ["item(call_a)", "response.create", "item(call_b)"]

        await _response_done(provider, session)

        assert _wire(ws) == [
            "item(call_a)",
            "response.create",
            "item(call_b)",
            "response.create",
        ]

    async def test_the_requested_response_owes_nothing_when_nothing_came_in_between(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)
        await _result(provider, session, "call_a")
        await _response_created(provider, session)

        await _response_done(provider, session)

        assert _wire(ws) == ["item(call_a)", "response.create"]


class TestAnEndedSession:
    @pytest.mark.parametrize("end", ["disconnect", "connection lost"])
    async def test_an_ended_session_keeps_no_open_calls(
        self, provider: OpenAIRealtimeBase, session: VoiceSession, end: str
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)

        if end == "disconnect":
            await provider.disconnect(session)
        else:
            await provider._retire_lost_connection(session, ws, "peer closed")

        assert session.id not in provider._pending_responses

    async def test_a_closed_session_sends_nothing(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)
        await provider.disconnect(session)
        sent_before = ws.send.await_count

        await _result(provider, session, "call_a")

        assert ws.send.await_count == sent_before
