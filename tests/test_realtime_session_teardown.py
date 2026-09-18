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
    assert not second.done(), "the second caller waits for the first"
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
