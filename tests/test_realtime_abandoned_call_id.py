"""A realtime call the provider abandons frees its id at once (RMK-460, RFC §12.4).

The provider frees the id when it reports the abandonment, while the handler
it interrupts may still be finishing (a cleanup that awaits). The channel
frees it at the same step: a call the vendor issues under the id meanwhile is
a new call, answered, and nothing is sent for the abandoned one. Before, the
channel held the id until the abandoned call's task ended, and refused the
new call as a duplicate the provider had booked: it was never answered.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from roomkit import ConferenceRealtimeConfig, HookExecution, HookTrigger, RoomKit
from roomkit.channels._realtime_tool_calls import RealtimeToolCall, ToolCallBook
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.test_realtime_own_reconnect import ReconnectingProvider, _channel

TOOLS = [{"name": "lookup", "description": "look up", "parameters": {"type": "object"}}]


class TestTheBook:
    @staticmethod
    def _running(session: Any) -> RealtimeToolCall:
        call = RealtimeToolCall(session, "c1", "lookup", {})
        call.task = SimpleNamespace(done=lambda: False)  # type: ignore[assignment]
        return call

    def test_an_abandoned_call_frees_its_id(self) -> None:
        book, session = ToolCallBook(), SimpleNamespace(id="s1")
        first = self._running(session)
        book.open(first)

        assert book.abandon("s1", "c1") is first
        second = RealtimeToolCall(session, "c1", "lookup", {})
        assert book.open(second)
        assert book.get("s1", "c1") is second and book.holds(first)

    def test_a_call_is_abandoned_once(self) -> None:
        book, session = ToolCallBook(), SimpleNamespace(id="s1")
        book.open(self._running(session))
        book.abandon("s1", "c1")

        assert book.abandon("s1", "c1") is None

    def test_a_delivered_call_is_not_abandoned(self) -> None:
        book, session = ToolCallBook(), SimpleNamespace(id="s1")
        call = self._running(session)
        book.open(call)
        call.delivered = True

        assert book.abandon("s1", "c1") is None
        assert not call.released


def _slow_cleanup_handler(*, swallow: bool = False) -> Any:
    """The first call waits, then awaits a cleanup when it is cancelled (or
    swallows the cancellation and answers); the next ones answer at once."""
    calls = 0

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        n = calls
        if n > 1:
            return f"found{n}"
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)  # closing a client
            if not swallow:
                raise
        return f"found{n}"

    return handler


async def _session(provider: MockRealtimeProvider, handler: Any) -> tuple[RoomKit, Any, list]:
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=TOOLS,
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    observed: list[tuple[str, bool, str]] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        observed.append((event.tool_call_id, event.cancelled, str(event.result)))

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    return kit, await channel.start_session("r1", "u", "ws"), observed


async def _reissue_during_cleanup(provider: MockRealtimeProvider, session: Any) -> None:
    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await asyncio.sleep(0.02)
    await provider.simulate_tool_call_cancellation(session, ["c1"])  # the provider frees c1
    await asyncio.sleep(0.01)
    await provider.simulate_tool_call(session, "c1", "lookup", {})  # the vendor issues c1 again


async def test_a_session_answers_an_id_reissued_while_the_abandoned_call_cleans_up() -> None:
    provider = MockRealtimeProvider()
    kit, session, observed = await _session(provider, _slow_cleanup_handler())

    await _reissue_during_cleanup(provider, session)
    await until(lambda: len(observed) == 2)
    await asyncio.sleep(0.1)
    await kit.close()

    assert [(r[1], r[2]) for r in provider.tool_results] == [("c1", "found2")]
    assert sorted((call_id, cancelled) for call_id, cancelled, _ in observed) == [
        ("c1", False),
        ("c1", True),
    ]


async def test_an_abandoned_call_that_answers_anyway_sends_nothing() -> None:
    """Its handler swallows the cancellation and answers: the id now names the
    new call, which must not read the abandoned call's answer."""
    provider = MockRealtimeProvider()
    kit, session, observed = await _session(provider, _slow_cleanup_handler(swallow=True))

    await _reissue_during_cleanup(provider, session)
    await until(lambda: len(observed) == 2)
    await asyncio.sleep(0.1)
    await kit.close()

    assert [(r[1], r[2]) for r in provider.tool_results] == [("c1", "found2")]


async def test_a_conference_answers_an_id_reissued_while_the_abandoned_call_cleans_up() -> None:
    provider = MockRealtimeProvider()
    handler = _slow_cleanup_handler()

    async def room_handler(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        return await handler(name, arguments)

    config = ConferenceRealtimeConfig(provider=provider, tools=TOOLS, tool_handler=room_handler)
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    session = await channel._realtime.ensure_session(ROOM)

    await _reissue_during_cleanup(provider, session)
    await until(lambda: len(provider.tool_results) == 1)
    await asyncio.sleep(0.1)
    await kit.close()

    assert [(r[1], r[2]) for r in provider.tool_results] == [("c1", "found2")]


async def test_the_new_socket_answers_the_id_of_the_call_whose_handler_reconnected() -> None:
    """The handler's own reconnect orphans its call, which runs on: the new
    socket never issued the id, and a call it issues under it is answered."""
    provider = ReconnectingProvider()
    holder: dict[str, Any] = {}
    reissued = asyncio.Event()

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        if name == "lookup":
            reissued.set()
            return "looked up"
        await holder["ch"].reconfigure_session(holder["session"], system_prompt="New.")
        await provider.inbox.put(("call", ("h1", "lookup", {})))
        await reissued.wait()
        await asyncio.sleep(0.02)
        return '{"accepted": true}'

    ch, [session], observed = await _channel(provider, handler)
    holder.update(ch=ch, session=session)

    await provider.simulate_tool_call(session, "h1", "switch_agent", {})
    await until(lambda: len(observed) == 2)
    for loop in provider.receive_loops:
        loop.cancel()

    # The spared call's result stays off the wire; the new call is answered.
    assert [(r[1], r[2]) for r in provider.tool_results] == [("h1", "looked up")]
    assert sorted((e.name, e.cancelled) for e in observed) == [
        ("lookup", False),
        ("switch_agent", False),
    ]
