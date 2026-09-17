"""A playback boundary remains observable behind continuous transport samples."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from roomkit.voice.realtime.pacer import OutboundAudioPacer


@pytest.mark.parametrize("new_response", [False, True])
async def test_boundary_ignores_following_silence_but_follows_new_eor(new_response):
    drains = [asyncio.Event(), asyncio.Event()]
    releases = [asyncio.Event(), asyncio.Event()]
    count = 0

    async def drain():
        nonlocal count
        index = count
        count += 1
        drains[index].set()
        await releases[index].wait()

    pacer = OutboundAudioPacer(AsyncMock(), 8000, fill_with_silence_when_idle=True)
    pacer._set_playback_observer(lambda *_: None, drain)
    await pacer.start()
    try:
        pacer.push(b"\x00\x20" * 160)
        pacer.end_of_response()
        await asyncio.wait_for(drains[0].wait(), 1)
        waiter = asyncio.create_task(pacer.wait_for_response_boundary())
        await asyncio.sleep(0)
        pacer.push(b"\x00\x00" * 160)
        if new_response:
            pacer.push(b"\x00\x20" * 160)
            pacer.end_of_response()
        # An expired observer cannot cancel the shared boundary.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(pacer.wait_for_response_boundary(), 0.01)
        releases[0].set()
        if new_response:
            await asyncio.wait_for(drains[1].wait(), 1)
            assert not waiter.done()
            releases[1].set()
        assert await asyncio.wait_for(waiter, 1)
    finally:
        for gate in releases:
            gate.set()
        await pacer.stop()


@pytest.mark.parametrize("stop", [False, True])
async def test_discarded_boundary_unblocks_with_failure(stop):
    pacer = OutboundAudioPacer(AsyncMock(), 8000)
    await pacer.start()
    pacer.push(b"\x00\x20" * 80000)
    pacer.end_of_response()
    waiter = asyncio.create_task(pacer.wait_for_response_boundary())
    await asyncio.sleep(0.01)
    try:
        if stop:
            await pacer.stop()
        else:
            pacer.interrupt()
        assert await asyncio.wait_for(waiter, 0.2) is False
    finally:
        await pacer.stop()


@pytest.mark.parametrize("samples", [160, 8000])
async def test_send_failure_settles_boundary_without_claiming_playback(samples):
    pacer = OutboundAudioPacer(AsyncMock(side_effect=OSError("transport closed")), 8000)
    await pacer.start()
    try:
        pacer.push(b"\x00\x20" * samples)
        pacer.end_of_response()
        assert await asyncio.wait_for(pacer.wait_for_response_boundary(), 2) is False
        assert not pacer._pending_boundaries
    finally:
        await pacer.stop()
