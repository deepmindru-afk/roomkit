"""Every realtime call reaches the channel, and every refusal at its entry
takes the path of any call (RMK-442, RFC §12.4).

A provider hands on a call that named no tool, which the channel refuses
before the gate and answers under its id. A call no result can name (no id,
an id in flight) is refused on the normal path: the session's end is read
first, a ``call_tool`` transport unwrapped, the transcription barrier waited
for, and nothing is sent. The ElevenLabs SDK cases live beside its canaries
(``tests/test_providers/test_elevenlabs_sdk_patch.py``).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import SecretStr

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.providers.deepgram.config import DeepgramAgentConfig
from roomkit.providers.deepgram.realtime import DeepgramAgentProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_realtime_call_intake import _gpt_live_calls
from tests.test_realtime_deepgram import _connect as deepgram_connect

NAMELESS = "Tool call named no tool"


def _session() -> VoiceSession:
    return VoiceSession(
        id="s1",
        room_id="r1",
        participant_id="p1",
        channel_id="rt",
        state=VoiceSessionState.CONNECTING,
    )


async def test_deepgram_hands_on_a_call_that_named_no_tool() -> None:
    provider = DeepgramAgentProvider(DeepgramAgentConfig(api_key=SecretStr("k")))
    session = _session()
    heard: list[Any] = []
    provider.on_tool_call(lambda *a: heard.append(a[1:3]))
    ws = await deepgram_connect(provider, session)
    ws.push(json.dumps({"type": "FunctionCallRequest", "functions": [{"id": "fc1"}]}))
    await asyncio.sleep(0.1)

    assert heard == [("fc1", "")]
    assert "fc1" in provider._states[session.id].pending_calls
    await provider.disconnect(session)


async def test_gpt_live_hands_on_a_call_that_named_no_tool() -> None:
    open_calls, heard = await _gpt_live_calls(
        [{"type": "function_call", "call_id": "c1", "arguments": "{}"}]
    )

    assert heard == [("c1", {})] and "c1" in open_calls


async def _channel(
    **kwargs: Any,
) -> tuple[RoomKit, RealtimeVoiceChannel, MockRealtimeProvider, Any, list[Any]]:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt", provider=provider, transport=MockRealtimeTransport(), **kwargs
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    seen: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        seen.append((event.tool_call_id, event.name, event.is_error, event.cancelled))

    session = await channel.start_session("r1", "u", "ws")
    return kit, channel, provider, session, seen


async def test_a_call_that_named_no_tool_is_refused_under_its_id() -> None:
    kit, _, provider, session, seen = await _channel(
        tools=[{"name": "t1", "parameters": {"type": "object"}}], tool_handler=lambda *a: "ok"
    )

    await provider.simulate_tool_call(session, "c1", None, {})  # type: ignore[arg-type]
    for _ in range(100):
        if provider.tool_results:
            break
        await asyncio.sleep(0.01)
    await kit.close()

    [(_, call_id, body)] = provider.tool_results
    assert call_id == "c1" and json.loads(body)["error"] == NAMELESS
    assert seen == [("c1", "", True, False)]


async def test_an_entry_refusal_waits_behind_the_barrier_under_the_wrapped_tool() -> None:
    names = [f"t{i}" for i in range(30)]
    kit, channel, provider, session, seen = await _channel(
        tools=[{"name": n, "description": n, "parameters": {"type": "object"}} for n in names],
        tool_handler=lambda *a: asyncio.sleep(0.2, "ok"),
        tool_search=True,
    )
    assert channel._tool_search_support is not None
    channel._tool_search_support.uses_call_tool = True
    wrapped = {"name": "t1", "arguments_json": "{}"}
    barrier = channel._transcription_order_locks.setdefault(session.id, asyncio.Lock())
    await barrier.acquire()

    await provider.simulate_tool_call(session, "d1", "call_tool", wrapped)
    await provider.simulate_tool_call(session, "d1", "call_tool", wrapped)
    await provider.simulate_tool_call(session, "", "call_tool", wrapped)
    await asyncio.sleep(0.1)
    reported_while_held = list(seen)
    barrier.release()
    for _ in range(100):
        if len(seen) == 3:
            break
        await asyncio.sleep(0.02)
    await kit.close()

    assert reported_while_held == []
    assert sorted(seen) == sorted(
        [("d1", "t1", True, False), ("", "t1", True, False), ("d1", "t1", False, False)]
    )
    # Only the call the id names was answered.
    assert [result[1] for result in provider.tool_results] == ["d1"]


async def test_an_id_less_call_on_an_ended_session_is_cancelled() -> None:
    kit, channel, provider, session, seen = await _channel(
        tools=[{"name": "t1", "parameters": {"type": "object"}}], tool_handler=lambda *a: "ok"
    )
    await channel.end_session(session)

    await provider.simulate_tool_call(session, "", "t1", {})
    await asyncio.sleep(0.1)
    await kit.close()

    assert seen == [("", "t1", True, True)]
    assert provider.tool_results == []
