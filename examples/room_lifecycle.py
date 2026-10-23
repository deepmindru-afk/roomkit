"""Room lifecycle — create, pause, close, timers.

Demonstrates the full room lifecycle with automatic timer-based
transitions. Shows:
- Creating rooms with RoomTimers (auto-pause, auto-close)
- Room statuses: ACTIVE -> PAUSED -> CLOSED
- check_room_timers() for timer evaluation
- Manual close_room()
- ON_ROOM_PAUSED and ON_ROOM_CLOSED lifecycle hooks

Run with:
    uv run python examples/room_lifecycle.py
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from roomkit import (
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomContext,
    RoomEvent,
    RoomKit,
    RoomTimers,
    TextContent,
    WebSocketChannel,
)

lifecycle_events: list[str] = []


async def main() -> None:
    kit = RoomKit()

    ws = WebSocketChannel("ws-user")
    kit.register_channel(ws)
    ws.register_connection("conn", lambda _c, _e: asyncio.sleep(0), room_id="room-manual")

    # --- Hook: Track lifecycle events ---
    @kit.hook(HookTrigger.ON_ROOM_PAUSED, execution=HookExecution.ASYNC, name="on_paused")
    async def on_paused(event: RoomEvent, ctx: RoomContext) -> None:
        lifecycle_events.append(f"PAUSED: Room {ctx.room.id}")

    @kit.hook(HookTrigger.ON_ROOM_CLOSED, execution=HookExecution.ASYNC, name="on_closed")
    async def on_closed(event: RoomEvent, ctx: RoomContext) -> None:
        lifecycle_events.append(f"CLOSED: Room {ctx.room.id}")

    # --- Room 1: Manual lifecycle ---
    print("=== Room 1: Manual lifecycle ===")
    room1 = await kit.create_room(room_id="room-manual", metadata={"topic": "Support"})
    print(f"  Created: status={room1.status}")

    await kit.attach_channel("room-manual", "ws-user")

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Hello!"),
        )
    )

    room1 = await kit.close_room("room-manual")
    print(f"  Closed: status={room1.status}")
    await asyncio.sleep(0.05)

    # --- Room 2: Timer-based auto-transitions ---
    print("\n=== Room 2: Timer-based lifecycle ===")

    # Create room with short timers for demo (normally minutes/hours). Until
    # the room has some activity, its inactivity counts from its creation.
    room2 = await kit.create_room(
        room_id="room-timed",
        timers=RoomTimers(
            inactive_after_seconds=5,  # Pause after 5s without activity
            closed_after_seconds=10,  # Close after 10s paused
        ),
    )
    await kit.attach_channel("room-timed", "ws-user")
    ws.subscribe("conn", "room-timed")
    print(f"  Created: status={room2.status}")

    # A delivered message is the room's activity: it moves last_activity_at.
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="I need help with my account"),
        )
    )

    # Backdate last_activity_at to simulate 6s of inactivity for the demo.
    await kit.set_room_timers(
        "room-timed",
        RoomTimers(
            inactive_after_seconds=5,
            closed_after_seconds=10,
            last_activity_at=datetime.now(UTC) - timedelta(seconds=6),  # 6s ago
        ),
    )

    # Check timers: PAUSED, its paused_at the instant it fell due (1s ago).
    room2 = await kit.check_room_timers("room-timed")
    print(f"  After 6s inactivity: status={room2.status}, paused_at={room2.paused_at}")
    await asyncio.sleep(0.05)

    # Simulate the room staying paused for 11s: the close timer counts from
    # paused_at. A host would simply call check_room_timers() later.
    await kit.store.update_room(
        room2.model_copy(update={"paused_at": datetime.now(UTC) - timedelta(seconds=11)})
    )
    room2 = await kit.check_room_timers("room-timed")
    print(f"  After 11s paused: status={room2.status}")
    await asyncio.sleep(0.05)

    # --- Room 3: check_all_timers for batch processing ---
    print("\n=== Room 3: Batch timer check ===")
    await kit.create_room(room_id="room-batch", timers=RoomTimers(inactive_after_seconds=2))
    await kit.attach_channel("room-batch", "ws-user")
    ws.subscribe("conn", "room-batch")
    # Backdate the last activity: the room has been idle for 3s (> 2s).
    await kit.set_room_timers(
        "room-batch",
        RoomTimers(
            inactive_after_seconds=2,
            last_activity_at=datetime.now(UTC) - timedelta(seconds=3),
        ),
    )

    transitioned = await kit.check_all_timers()
    print(f"  Batch check: {len(transitioned)} room(s) transitioned")
    for r in transitioned:
        print(f"    {r.id}: -> {r.status}")

    await asyncio.sleep(0.1)

    # --- Show lifecycle events ---
    print(f"\nLifecycle events ({len(lifecycle_events)}):")
    for entry in lifecycle_events:
        print(f"  {entry}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
