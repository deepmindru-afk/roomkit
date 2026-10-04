"""The ElevenLabs SDK patch and the canary that says when to drop it (RMK-440).

The SDK's ``ClientTools`` answers a call to an unregistered name itself;
``providers/elevenlabs/sdk_patch.py`` hands such a call to the channel. The
canary fails once the SDK stops answering it itself, or stops dispatching
through ``handle``: the patch is then dropped, or reworked.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from roomkit.providers.elevenlabs import sdk_patch
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState

conversation = pytest.importorskip("elevenlabs.conversational_ai.conversation")


def _session() -> VoiceSession:
    return VoiceSession(
        id="s1",
        room_id="r1",
        participant_id="p1",
        channel_id="rt",
        state=VoiceSessionState.CONNECTING,
    )


async def _execute(tools: Any, name: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    tools.start()
    try:
        tools.execute_tool(name, {"tool_call_id": "t1"}, results.append)
        for _ in range(100):
            if results:
                break
            await asyncio.sleep(0.01)
    finally:
        tools.stop()
    [result] = results
    return result


async def test_the_sdk_still_answers_an_unregistered_tool_itself() -> None:
    tools = conversation.ClientTools(loop=asyncio.get_running_loop())

    result = await _execute(tools, "secret_op")

    assert result["is_error"] is True
    assert "not registered" in result["result"]


async def test_the_patch_routes_an_unregistered_tool_through_the_sdks_dispatch() -> None:
    routed: list[str] = []

    async def route(name: str, parameters: dict[str, Any]) -> str:
        routed.append(name)
        return "refused by the channel"

    tools = sdk_patch.client_tools(
        conversation.ClientTools, loop=asyncio.get_running_loop(), route=route
    )

    result = await _execute(tools, "secret_op")

    assert routed == ["secret_op"]
    assert (result["result"], result["is_error"]) == ("refused by the channel", False)


async def test_the_sdk_sends_nothing_for_a_cancelled_handler() -> None:
    """How the provider sends nothing for a call no result can name."""
    tools = conversation.ClientTools(loop=asyncio.get_running_loop())

    async def unanswerable(parameters: dict[str, Any]) -> str:
        raise asyncio.CancelledError

    tools.register("lookup", unanswerable, is_async=True)
    results: list[dict[str, Any]] = []
    tools.start()
    try:
        tools.execute_tool("lookup", {"tool_call_id": "t1"}, results.append)
        await asyncio.sleep(0.1)
    finally:
        tools.stop()

    assert results == []


# -- RMK-442: the service's id, whatever the model wrote ---------------------


def _patched_elevenlabs() -> tuple[ElevenLabsRealtimeProvider, Any, list[Any], list[Any]]:
    """The provider behind the SDK's own message core, patched as the
    provider builds it: what the channel hears and what goes on the wire."""
    provider = ElevenLabsRealtimeProvider(ElevenLabsRealtimeConfig(api_key="k", agent_id="a"))
    session = _session()
    heard: list[Any] = []

    def on_call(s: Any, call_id: str, name: Any, arguments: Any) -> None:
        heard.append((call_id, name, arguments))
        asyncio.get_running_loop().create_task(provider.submit_tool_result(s, call_id, "ok"))

    provider.on_tool_call(on_call)
    tools = sdk_patch.client_tools(
        conversation.ClientTools,
        loop=asyncio.get_running_loop(),
        route=lambda name, params: provider._route_unregistered(session, name, params),
    )
    tools.start()
    patched_class = sdk_patch.conversation(conversation.AsyncConversation)
    patched = patched_class.__new__(patched_class)
    patched.client_tools = tools
    sent: list[Any] = []
    handler = SimpleNamespace(
        handle_client_tool_call=lambda name, params: tools.execute_tool(name, params, sent.append)
    )
    return provider, (patched, handler), heard, sent


async def _wire(patched: Any, handler: Any, call: dict[str, Any]) -> None:
    message = {"type": "client_tool_call", "client_tool_call": call}
    await patched._handle_message_core_async(message, handler)
    await asyncio.sleep(0.1)


async def test_elevenlabs_hands_on_a_call_that_named_no_tool() -> None:
    _, (patched, handler), heard, _ = _patched_elevenlabs()

    await _wire(patched, handler, {"tool_call_id": "real-1", "parameters": {"q": 1}})

    assert heard[0][:2] == ("real-1", "")


async def test_elevenlabs_answers_under_the_service_s_id() -> None:
    """A ``tool_call_id`` the model wrote is one of its arguments, never the
    call's id (decision A): the channel and the wire both use the service's."""
    _, (patched, handler), heard, sent = _patched_elevenlabs()

    await _wire(
        patched,
        handler,
        {"tool_call_id": "real-2", "tool_name": "lookup", "parameters": {"tool_call_id": "x"}},
    )

    assert heard == [("real-2", "lookup", {"tool_call_id": "x"})]
    assert [response["tool_call_id"] for response in sent] == ["real-2"]


async def test_elevenlabs_survives_a_call_without_an_id() -> None:
    """The SDK raised KeyError, which ended the conversation; the call now
    reaches the channel without an id, which refuses it."""
    _, (patched, handler), heard, _ = _patched_elevenlabs()

    await _wire(patched, handler, {"tool_name": "lookup"})

    assert heard[0][:2] == ("", "lookup")


async def test_the_sdk_still_lets_a_parameter_replace_the_call_id() -> None:
    """Canary: when this fails, the SDK keeps the service's id itself, and
    ``sdk_patch.conversation`` and ``split_call`` go (see the module)."""
    received: list[dict[str, Any]] = []
    handler = SimpleNamespace(handle_client_tool_call=lambda name, params: received.append(params))
    message = {
        "type": "client_tool_call",
        "client_tool_call": {
            "tool_call_id": "real",
            "tool_name": "lookup",
            "parameters": {"tool_call_id": "forged"},
        },
    }

    await conversation.AsyncConversation._handle_message_core_async(
        SimpleNamespace(), message, handler
    )

    assert received[0]["tool_call_id"] == "forged"


async def test_elevenlabs_hands_on_parameters_that_are_no_object_as_text() -> None:
    """The SDK's ``**parameters`` raised on them, which ended the
    conversation; they reach the channel as the model's text, which it
    refuses as unreadable."""
    _, (patched, handler), heard, sent = _patched_elevenlabs()

    await _wire(
        patched, handler, {"tool_call_id": "real-3", "tool_name": "lookup", "parameters": [1, 2]}
    )

    assert heard == [("real-3", "lookup", "[1, 2]")]
    assert [response["tool_call_id"] for response in sent] == ["real-3"]


@pytest.mark.parametrize(
    ("parameters", "arguments"),
    [('{"q": "a"}', {"q": "a"}), ("", {}), ("null", {}), ("  ", {}), ("[1]", "[1]")],
)
async def test_elevenlabs_reads_text_parameters_as_every_realtime_provider(
    parameters: str, arguments: Any
) -> None:
    """The shared rule (``readable_arguments``): text that reads as an object
    is that object, no arguments are ``{}``, anything else stays text."""
    _, (patched, handler), heard, _ = _patched_elevenlabs()

    await _wire(
        patched, handler, {"tool_call_id": "c1", "tool_name": "lookup", "parameters": parameters}
    )

    assert heard == [("c1", "lookup", arguments)]


class _Socket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))


async def test_the_sdks_own_message_handling_goes_through_the_patch() -> None:
    """Through the SDK's ``_handle_message``, the entry its socket loop calls:
    an SDK that stopped routing through ``_handle_message_core_async`` would
    hand the model's ``tool_call_id`` over as the call's id again."""
    provider, (patched, _), heard, _ = _patched_elevenlabs()
    session = _session()
    provider._register_client_tools(patched.client_tools, session, [{"name": "lookup"}])
    patched._should_stop = threading.Event()
    patched._conversation_id = None
    patched._last_interrupt_id = 0
    patched.audio_interface = None
    for callback in (
        "callback_agent_response",
        "callback_agent_response_correction",
        "callback_agent_chat_response_part",
        "callback_user_transcript",
        "callback_latency_measurement",
        "callback_audio_alignment",
    ):
        setattr(patched, callback, None)
    socket = _Socket()
    call = {"tool_call_id": "real-4", "tool_name": "lookup", "parameters": {"tool_call_id": "x"}}

    await patched._handle_message({"type": "client_tool_call", "client_tool_call": call}, socket)
    await asyncio.sleep(0.15)

    assert heard == [("real-4", "lookup", {"tool_call_id": "x"})]
    assert [message.get("tool_call_id") for message in socket.sent] == ["real-4"]
