"""Deferred connections keep both halves of startup under one session owner."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport


class ControlledProvider(MockRealtimeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.ready = asyncio.Event()
        self.release = asyncio.Event()
        self.session: VoiceSession | None = None

    async def connect(self, session: VoiceSession, **kwargs: Any) -> None:
        self.session = session
        self.entered.set()
        await self.release.wait()
        await super().connect(session, **kwargs)
        self.ready.set()


@pytest.mark.parametrize("answer_first", [False, True])
async def test_join_overlaps_setup_but_publishes_only_when_both_ready(answer_first: bool) -> None:
    provider = ControlledProvider()
    transport = MockRealtimeTransport()
    channel = RealtimeVoiceChannel("rt", provider=provider, transport=transport)
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "rt")
    connection = asyncio.get_running_loop().create_future()
    start = asyncio.create_task(kit.join("room", "rt", connection=connection))
    await provider.entered.wait()
    assert provider.session is not None
    if answer_first:
        connection.set_result("carrier")
        while not transport._connections:
            await asyncio.sleep(0)
        await transport.simulate_client_audio(provider.session, b"first word")
        assert not provider.sent_audio
    else:
        provider.release.set()
        await provider.ready.wait()
        assert not transport._connections
    assert not channel.get_room_sessions("room")
    assert not start.done()
    if answer_first:
        provider.release.set()
    else:
        connection.set_result("carrier")
    session = await asyncio.wait_for(start, 1)
    assert [c.method for c in provider.calls].count("connect") == 1
    assert [c.method for c in transport.calls].count("accept") == 1
    timing = session.metadata["connection_timing"]
    assert (timing["provider_ready_at"] > timing["transport_ready_at"]) == answer_first
    if answer_first:
        assert provider.sent_audio == [(session.id, b"first word")]
    await kit.close()


@pytest.mark.parametrize("provider_ready", [False, True])
@pytest.mark.parametrize("end", ["cancel", "close", "connection_failure"])
async def test_abandoned_preparation_releases_both_branches(
    provider_ready: bool, end: str
) -> None:
    provider = ControlledProvider()
    transport = MockRealtimeTransport()
    channel = RealtimeVoiceChannel("rt", provider=provider, transport=transport)
    connection = asyncio.get_running_loop().create_future()
    start = asyncio.create_task(channel.start_session("r", "p", connection))
    await provider.entered.wait()
    if provider_ready:
        provider.release.set()
        await provider.ready.wait()
    if end == "cancel":
        start.cancel()
    elif end == "close":
        await channel.close()
    else:
        connection.set_exception(RuntimeError("rejected"))
    with pytest.raises((asyncio.CancelledError, RuntimeError)):
        await asyncio.wait_for(start, 1)
    assert not channel._connecting_sessions
    assert not channel.get_room_sessions("r")
    assert not channel._preconnect_audio
    assert [c.method for c in provider.calls].count("disconnect") == 1
    assert [c.method for c in transport.calls].count("disconnect") == 1
    await channel.close()


async def test_provider_failure_cancels_connection_wait_without_leaking_a_session() -> None:
    class FailedProvider(MockRealtimeProvider):
        async def connect(self, session: VoiceSession, **kwargs: Any) -> None:
            raise RuntimeError("provider failed")

    transport = MockRealtimeTransport()
    channel = RealtimeVoiceChannel("rt", provider=FailedProvider(), transport=transport)
    connection = asyncio.get_running_loop().create_future()
    with pytest.raises(RuntimeError, match="provider failed"):
        await asyncio.wait_for(channel.start_session("r", "p", connection), 1)
    assert connection.cancelled()
    assert not transport._connections
    assert not channel._connecting_sessions
    await channel.close()


async def test_bye_during_provider_preparation_cannot_activate_the_session() -> None:
    provider = ControlledProvider()
    transport = MockRealtimeTransport()
    channel = RealtimeVoiceChannel("rt", provider=provider, transport=transport)
    connection = asyncio.get_running_loop().create_future()
    start = asyncio.create_task(channel.start_session("r", "p", connection))
    await provider.entered.wait()
    connection.set_result("carrier")
    while not transport._connections:
        await asyncio.sleep(0)
    assert provider.session is not None
    await transport.simulate_client_disconnect(provider.session)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start, 1)
    provider.release.set()
    assert not channel.get_room_sessions("r")
    assert not transport._connections
    await channel.close()


async def test_simultaneous_preparations_never_exchange_participants_or_audio() -> None:
    provider = MockRealtimeProvider()
    transport = MockRealtimeTransport()
    channel = RealtimeVoiceChannel("rt", provider=provider, transport=transport)
    loop = asyncio.get_running_loop()
    first, second = loop.create_future(), loop.create_future()
    one = asyncio.create_task(channel.start_session("room-a", "peer-a", first))
    two = asyncio.create_task(channel.start_session("room-b", "peer-b", second))
    second.set_result("connection-b")
    session_b = await asyncio.wait_for(two, 1)
    await transport.simulate_client_audio(session_b, b"from b")
    first.set_result("connection-a")
    session_a = await asyncio.wait_for(one, 1)
    await transport.simulate_client_audio(session_a, b"from a")
    await asyncio.sleep(0)
    assert session_a.id != session_b.id
    assert transport._connections == {
        session_a.id: "connection-a",
        session_b.id: "connection-b",
    }
    assert provider.sent_audio == [(session_b.id, b"from b"), (session_a.id, b"from a")]
    await channel.end_session(session_b)
    assert channel.get_room_sessions("room-a") == [session_a]
    await channel.close()
