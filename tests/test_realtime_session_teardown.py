"""A realtime session is torn down once; callers that arrive during the
teardown join it."""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import RealtimeVoiceChannel
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport


class _SlowDisconnectProvider(MockRealtimeProvider):
    """A provider whose disconnect waits on the peer, the way a live socket does."""

    def __init__(self) -> None:
        super().__init__()
        self.disconnecting = asyncio.Event()
        self.release = asyncio.Event()

    async def disconnect(self, session: VoiceSession) -> None:
        self.disconnecting.set()
        await self.release.wait()
        await super().disconnect(session)


async def _started() -> tuple[
    RealtimeVoiceChannel, _SlowDisconnectProvider, MockRealtimeTransport, Any
]:
    provider = _SlowDisconnectProvider()
    transport = MockRealtimeTransport()
    channel = RealtimeVoiceChannel("rt", provider=provider, transport=transport)
    session = await channel.start_session("r", "p", "fake-ws")
    return channel, provider, transport, session


def _disconnects(calls: list[Any]) -> int:
    return [c.method for c in calls].count("disconnect")


async def test_concurrent_end_session_calls_share_one_teardown() -> None:
    channel, provider, transport, session = await _started()

    first = asyncio.create_task(channel.end_session(session))
    await provider.disconnecting.wait()
    second = asyncio.create_task(channel.end_session(session))
    await asyncio.sleep(0)
    # The second caller has reached the wait without starting a disconnect of
    # its own: the provider still records none while the first waits on it.
    assert not second.done() and _disconnects(provider.calls) == 0
    provider.release.set()
    await asyncio.gather(first, second)

    assert _disconnects(provider.calls) == 1
    assert _disconnects(transport.calls) == 1
    assert not channel.get_room_sessions("r")
    assert not channel._session_teardowns
    await channel.close()


async def test_close_joins_the_teardown_the_client_disconnect_started() -> None:
    """The remote hangup: the transport's callback ends the session, and the
    application's ``close()`` arrives while that teardown waits on the
    provider. One ``provider.disconnect``, one ``transport.disconnect``."""
    channel, provider, transport, session = await _started()

    await transport.simulate_client_disconnect(session)
    await provider.disconnecting.wait()
    closing = asyncio.create_task(channel.close())
    await asyncio.sleep(0)
    assert not closing.done()
    provider.release.set()
    await closing

    assert _disconnects(provider.calls) == 1
    assert _disconnects(transport.calls) == 1
    assert not channel.get_room_sessions("r")


async def test_a_joiner_cancelled_while_waiting_leaves_the_owner_untouched() -> None:
    channel, provider, transport, session = await _started()

    owner = asyncio.create_task(channel.end_session(session))
    await provider.disconnecting.wait()
    joiner = asyncio.create_task(channel.end_session(session))
    await asyncio.sleep(0)
    joiner.cancel()
    await asyncio.gather(joiner, return_exceptions=True)
    assert joiner.cancelled()
    provider.release.set()
    await owner

    assert _disconnects(provider.calls) == 1
    assert _disconnects(transport.calls) == 1
    assert not channel.get_room_sessions("r")
    await channel.close()


class _FailingBeforeTeardown(RealtimeVoiceChannel):
    """A subclass teardown that fails, once a joiner has had time to arrive."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def _before_session_teardown(self, session: VoiceSession) -> None:
        self.entered.set()
        await self.release.wait()
        raise RuntimeError("the hook failed")


async def test_an_owner_that_fails_still_releases_its_joiners() -> None:
    """A joiner learns that the teardown ended, not how: the owner's exception
    is the owner's, and the registry is clear for the next caller."""
    provider = _SlowDisconnectProvider()
    transport = MockRealtimeTransport()
    channel = _FailingBeforeTeardown("rt", provider=provider, transport=transport)
    session = await channel.start_session("r", "p", "fake-ws")

    owner = asyncio.create_task(channel.end_session(session))
    await channel.entered.wait()
    joiner = asyncio.create_task(channel.end_session(session))
    await asyncio.sleep(0)
    channel.release.set()
    results = await asyncio.gather(owner, joiner, return_exceptions=True)

    assert isinstance(results[0], RuntimeError) and results[1] is None
    assert not channel._session_teardowns
    provider.release.set()
    await channel.close()


class _ReentrantBeforeTeardown(RealtimeVoiceChannel):
    """A handler told the session ended ends it again, from the owner's task."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reentered = 0

    async def _before_session_teardown(self, session: VoiceSession) -> None:
        self.reentered += 1
        await self.end_session(session)


async def test_a_call_reentered_from_the_teardown_returns_at_once() -> None:
    provider = _SlowDisconnectProvider()
    provider.release.set()
    transport = MockRealtimeTransport()
    channel = _ReentrantBeforeTeardown("rt", provider=provider, transport=transport)
    session = await channel.start_session("r", "p", "fake-ws")

    await asyncio.wait_for(channel.end_session(session), timeout=2)

    assert channel.reentered == 1
    assert _disconnects(provider.calls) == 1
    assert not channel._session_teardowns
    await channel.close()


async def test_the_av_channel_runs_its_own_teardown_once_too() -> None:
    """The arbitration covers a subclass's teardown: the video hooks ride the
    owner, so a remote hangup reaching the AV channel twice fires them once."""
    from roomkit.channels.realtime_av import RealtimeAudioVideoChannel

    provider = _SlowDisconnectProvider()
    transport = MockRealtimeTransport()
    channel = RealtimeAudioVideoChannel("av", provider=provider, transport=transport)
    session = await channel.start_session("r", "p", "fake-ws")
    before = channel._before_session_teardown
    ran: list[str] = []

    async def counted(ended: VoiceSession) -> None:
        ran.append(ended.id)
        await before(ended)

    channel._before_session_teardown = counted  # type: ignore[method-assign]

    first = asyncio.create_task(channel.end_session(session))
    await provider.disconnecting.wait()
    second = asyncio.create_task(channel.end_session(session))
    await asyncio.sleep(0)
    provider.release.set()
    await asyncio.gather(first, second)

    assert ran == [session.id]
    assert _disconnects(provider.calls) == 1
    await channel.close()
