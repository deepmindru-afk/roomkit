"""Prepare a realtime provider while waiting for a client connection.

Run with: uv run python examples/realtime_deferred_connection.py
Uses mock peers; no credentials, external calls or audio hardware required.
"""

from __future__ import annotations

import asyncio
import logging

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

logger = logging.getLogger(__name__)


async def main() -> None:
    kit = RoomKit()
    channel = RealtimeVoiceChannel(
        "voice", provider=MockRealtimeProvider(), transport=MockRealtimeTransport()
    )
    kit.register_channel(channel)
    await kit.create_room(room_id="call")
    await kit.attach_channel("call", "voice")
    connection: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    join = asyncio.create_task(kit.join("call", "voice", connection=connection))
    try:
        # The call owner bounds ringing. In SIP, resolve with the answered
        # VoiceSession; a WebSocket application can resolve with its socket.
        async with asyncio.timeout(2):
            await asyncio.sleep(0.1)
            connection.set_result("mock-client")
            session = await join
        logger.info("Session ready: %s", session.metadata["connection_timing"])
        await kit.leave(session)
    finally:
        join.cancel()
        await asyncio.gather(join, return_exceptions=True)
        await kit.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
