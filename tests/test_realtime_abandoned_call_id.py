"""A realtime call the provider abandons frees its id at once (RMK-460, RFC §12.4).

The provider frees the id when it reports the abandonment, while the handler
it interrupts may still be finishing (a cleanup that awaits), or while its
observers still hear its outcome. The channel and the conference free it at
the same step: a call the vendor issues under the id meanwhile is a new call,
answered, and nothing is sent for the abandoned one.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

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

        assert book.release("s1", "c1") is first and first.interruptible
        second = RealtimeToolCall(session, "c1", "lookup", {})
        assert book.open(second)
        assert book.get("s1", "c1") is second and book.holds(first)

    def test_a_call_is_released_once(self) -> None:
        book, session = ToolCallBook(), SimpleNamespace(id="s1")
        book.open(self._running(session))
        book.release("s1", "c1")

        assert book.release("s1", "c1") is None

    def test_a_delivered_call_is_not_released(self) -> None:
        book, session = ToolCallBook(), SimpleNamespace(id="s1")
        call = self._running(session)
        book.open(call)
        call.delivered = True

        assert book.release("s1", "c1") is None
        assert not call.released

    def test_a_reported_call_is_released_but_not_interrupted(self) -> None:
        book, session = ToolCallBook(), SimpleNamespace(id="s1")
        call = self._running(session)
        book.open(call)
        call.reported = True

        assert book.release("s1", "c1") is call
        assert call.released and not call.interruptible


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


async def _door(
    door: str, handler: Any, hold_first_report: asyncio.Event | None = None
) -> tuple[RoomKit, MockRealtimeProvider, Any, list[tuple[str, bool]]]:
    """A session on the realtime channel or on a conference, its tool calls
    served by *handler*, and what ON_TOOL_CALL's observers hear. The first
    report waits on *hold_first_report* when given: an audit writing it."""
    provider = MockRealtimeProvider()
    if door == "session":
        channel = RealtimeVoiceChannel(
            "rt",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=TOOLS,
            tool_handler=handler,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "rt")
        session = await channel.start_session("r1", "u", "ws")
    else:

        async def room_handler(room_id: str, name: str, arguments: dict[str, Any]) -> str:
            return await handler(name, arguments)

        config = ConferenceRealtimeConfig(
            provider=provider, tools=TOOLS, tool_handler=room_handler
        )
        kit, conference, _, _ = await realtime_kit(provider=provider, config=config)
        session = await conference._realtime.ensure_session(ROOM)
    observed: list[tuple[str, bool]] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        observed.append((event.tool_call_id, event.cancelled))
        if hold_first_report is not None and len(observed) == 1:
            await hold_first_report.wait()

    return kit, provider, session, observed


async def _reissue_during_cleanup(provider: MockRealtimeProvider, session: Any) -> None:
    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await asyncio.sleep(0.02)
    await provider.simulate_tool_call_cancellation(session, ["c1"])  # the provider frees c1
    await asyncio.sleep(0.01)
    await provider.simulate_tool_call(session, "c1", "lookup", {})  # the vendor issues c1 again


DOORS = pytest.mark.parametrize("door", ["session", "conference"])


@DOORS
async def test_an_id_reissued_while_the_abandoned_call_cleans_up_is_answered(door: str) -> None:
    kit, provider, session, observed = await _door(door, _slow_cleanup_handler())

    await _reissue_during_cleanup(provider, session)
    await until(lambda: len(observed) == 2)
    await asyncio.sleep(0.1)
    await kit.close()

    assert [(r[1], r[2]) for r in provider.tool_results] == [("c1", "found2")]
    assert sorted(observed) == [("c1", False), ("c1", True)]


@DOORS
async def test_an_abandoned_call_that_answers_anyway_sends_nothing(door: str) -> None:
    """Its handler swallows the cancellation and answers: the id names the new
    call, which must not read the abandoned call's answer."""
    kit, provider, session, observed = await _door(door, _slow_cleanup_handler(swallow=True))

    await _reissue_during_cleanup(provider, session)
    await until(lambda: len(observed) == 2)
    await asyncio.sleep(0.1)
    await kit.close()

    assert [(r[1], r[2]) for r in provider.tool_results] == [("c1", "found2")]
    assert sorted(observed) == [("c1", False), ("c1", True)]


@DOORS
async def test_an_id_reissued_while_the_observers_hear_the_call_is_answered(door: str) -> None:
    """The provider abandons a call whose observers already heard it, before
    its result went out: nothing is sent for it, and the call issued under its
    id gets its own answer."""
    calls = 0

    async def counting(name: str, arguments: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        return f"found{calls}"

    release = asyncio.Event()
    kit, provider, session, observed = await _door(door, counting, hold_first_report=release)

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: len(observed) == 1)  # the audit writes "found1"
    await provider.simulate_tool_call_cancellation(session, ["c1"])
    await provider.simulate_tool_call(session, "c1", "lookup", {})
    release.set()
    await until(lambda: len(observed) == 2)
    await asyncio.sleep(0.1)
    await kit.close()

    assert [(r[1], r[2]) for r in provider.tool_results] == [("c1", "found2")]
    assert observed == [("c1", False), ("c1", False)]


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
