"""A realtime call its response cut runs only when its argument text reads
(RMK-455, RFC §6.4, §12.4).

The rule the text providers apply (``partial_when``) holds for the realtime
providers that can tell a cut: OpenAI Realtime and xAI read the item's status
once it is done (``function_call_arguments.done`` comes before it), GPT-Live
reads its backend's item. A call cut before its arguments were whole reaches
the channel as :class:`CutArguments`, refused before the gate as cut off;
whole argument text under a cut runs. Deepgram's requests carry no sign of a
cut.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.providers.ai.tool_calls import CutArguments, realtime_call_arguments
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_providers.test_openai_realtime_tool_results import _PROVIDERS, _attach
from tests.test_realtime_call_intake import _gpt_live_calls

CUT_OFF = "Tool call cut off"


@pytest.mark.parametrize(
    ("raw", "cut", "expected"),
    [
        ("", True, CutArguments("")),
        ('{"note": "Paris, the', True, CutArguments('{"note": "Paris, the')),
        ("[1, 2]", True, CutArguments("[1, 2]")),
        ('{"note": "short"}', True, {"note": "short"}),
        ("", False, {}),
        ('{"note": "Paris, the', False, '{"note": "Paris, the'),
    ],
)
def test_the_shared_rule(raw: str, cut: bool, expected: Any) -> None:
    read = realtime_call_arguments(raw, cut=cut)

    assert read == expected
    assert isinstance(read, CutArguments) == isinstance(expected, CutArguments)


def _item(status: str, arguments: str) -> dict[str, Any]:
    return {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "call_id": "c1",
            "name": "save_note",
            "arguments": arguments,
            "status": status,
        },
    }


@pytest.mark.parametrize("vendor", sorted(_PROVIDERS))
async def test_openai_realtime_hands_a_call_on_once_its_item_says_whether_it_was_cut(
    vendor: str,
) -> None:
    """Measured on the wire (OpenAI Realtime, ``max_output_tokens`` 40):
    ``function_call_arguments.done`` with the cut text, then the item done
    with ``status: incomplete``, then ``response.done`` incomplete."""
    provider = _PROVIDERS[vendor]()
    session = VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="voice")
    _attach(provider, session)
    heard: list[Any] = []
    provider.on_tool_call(lambda *a: heard.append(a[3]))
    cut_text = '{"title":"Paris","note":"Paris, the capital of France, has a'

    await provider._handle_server_event(
        session,
        {
            "type": "response.function_call_arguments.done",
            "call_id": "c1",
            "name": "save_note",
            "arguments": cut_text,
        },
    )
    assert heard == []
    await provider._handle_server_event(session, _item("incomplete", cut_text))
    await provider._handle_server_event(session, _item("incomplete", ""))
    await provider._handle_server_event(session, _item("completed", "{}"))

    assert heard == [CutArguments(cut_text), CutArguments(""), {}]
    assert [isinstance(a, CutArguments) for a in heard] == [True, True, False]


async def test_gpt_live_refuses_a_cut_call_with_no_argument_text() -> None:
    item = {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": ""}
    _, heard = await _gpt_live_calls([{**item, "status": "incomplete"}])

    [(call_id, arguments)] = heard
    assert call_id == "c1" and isinstance(arguments, CutArguments)


async def test_the_channel_refuses_a_cut_call_as_cut_off() -> None:
    provider = MockRealtimeProvider()
    ran: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "ok"

    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[{"name": "save_note", "parameters": {"type": "object"}}],
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    seen: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        seen.append((event.is_error, json.loads(str(event.result))["error"]))

    session = await channel.start_session("r1", "u", "ws")
    await provider.simulate_tool_call(session, "c1", "save_note", CutArguments(""))
    for _ in range(100):
        if seen and provider.tool_results:
            break
        await asyncio.sleep(0.01)
    await kit.close()

    assert ran == []
    assert json.loads(provider.tool_results[0][2])["error"] == CUT_OFF
    assert seen == [(True, CUT_OFF)]
