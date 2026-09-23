"""A slow content check that runs off the room lock, and fails closed.

A BEFORE_BROADCAST check that calls a remote service (a PII scanner, a
moderation API) holds the room lock for its whole duration by default: the
messages of one room queue behind each other's scans. ``needs_lock=False``
runs the check before the lock is taken (RFC §9.5.1) — scans overlap, and a
per-room admission ticket keeps the messages in arrival order.
``fail_closed=True`` blocks a message whose check timed out or failed instead
of delivering it unchecked (RFC §9.3).

The scanner is simulated with a 1 s sleep. Two messages sent 0.2 s apart go
out at about 1.0 s and 1.2 s; with the check under the lock (drop
``needs_lock=False``) the second one would wait until about 2.0 s.

Run with:
    uv run python examples/hook_off_lock_check.py
"""

from __future__ import annotations

import asyncio

from shared import setup_logging

from roomkit import (
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)

logger = setup_logging("hook_off_lock_check")

SCAN_SECONDS = 1.0


async def main() -> None:
    kit = RoomKit()
    ws_user = WebSocketChannel("ws-user")
    ws_agent = WebSocketChannel("ws-agent")
    kit.register_channel(ws_user)
    kit.register_channel(ws_agent)

    loop = asyncio.get_running_loop()
    start = loop.time()

    async def agent_recv(_conn: str, event: RoomEvent) -> None:
        body = event.content.body if isinstance(event.content, TextContent) else ""
        logger.info("t=%.1fs delivered: %s", loop.time() - start, body)

    ws_agent.register_connection("agent-conn", agent_recv, room_id="room")
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "ws-user")
    await kit.attach_channel("room", "ws-agent")

    @kit.hook(
        HookTrigger.BEFORE_BROADCAST,
        name="pii_scan",
        timeout=2.0,
        fail_closed=True,
        needs_lock=False,
    )
    async def pii_scan(event: RoomEvent, ctx: RoomContext) -> HookResult:
        body = event.content.body if isinstance(event.content, TextContent) else ""
        # Stand-in for the scanner's HTTP round trip; "hang" never answers.
        await asyncio.sleep(10 if body == "hang" else SCAN_SECONDS)
        return HookResult.allow()

    async def send(body: str, delay: float) -> None:
        await asyncio.sleep(delay)
        result = await kit.process_inbound(
            InboundMessage(channel_id="ws-user", sender_id="user", content=TextContent(body=body)),
            room_id="room",
        )
        if result.blocked:
            logger.info("t=%.1fs blocked: %s (%s)", loop.time() - start, body, result.reason)

    # Two messages 0.2 s apart: their scans overlap, their order holds.
    await asyncio.gather(send("first", 0.0), send("second", 0.2))

    # A scanner that never answers: the message is blocked, not delivered.
    start = loop.time()
    await send("hang", 0.0)


if __name__ == "__main__":
    asyncio.run(main())
