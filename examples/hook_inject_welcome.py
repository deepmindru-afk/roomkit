"""Auto-inject welcome messages with hooks.

Demonstrates how hooks greet each participant who joins a room:

- Lifecycle hooks (ON_ROOM_CREATED, ON_CHANNEL_ATTACHED): the attach hook
  notes each newcomer.  Lifecycle hooks are observers — their return value
  is ignored — so they cannot inject anything themselves.
- InjectedEvent from a sync BEFORE_BROADCAST hook: when a newcomer's first
  message goes through, the hook lets it pass and injects a welcome
  delivered to that newcomer only (``target_channel_ids``).
- SystemContent for the welcome notification.

Run with:
    uv run python examples/hook_inject_welcome.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    ChannelType,
    EventSource,
    EventType,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    InjectedEvent,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.models.event import SystemContent


def _welcome(room_id: str, channel_id: str) -> InjectedEvent:
    """A system welcome addressed to *channel_id* only."""
    return InjectedEvent(
        event=RoomEvent(
            room_id=room_id,
            type=EventType.SYSTEM,
            source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
            content=SystemContent(
                body=f"Welcome to {room_id}, {channel_id}!",
                code="welcome",
                data={"channel_id": channel_id},
            ),
        ),
        target_channel_ids=[channel_id],
    )


async def main() -> None:
    kit = RoomKit()

    ws_alice = WebSocketChannel("ws-alice")
    ws_bob = WebSocketChannel("ws-bob")
    kit.register_channel(ws_alice)
    kit.register_channel(ws_bob)

    alice_inbox: list[RoomEvent] = []
    bob_inbox: list[RoomEvent] = []

    async def alice_recv(_conn: str, event: RoomEvent) -> None:
        alice_inbox.append(event)

    async def bob_recv(_conn: str, event: RoomEvent) -> None:
        bob_inbox.append(event)

    ws_alice.register_connection("alice-conn", alice_recv, room_id="welcome-room")
    ws_bob.register_connection("bob-conn", bob_recv, room_id="welcome-room")

    # Channels attached but not greeted yet.
    newcomers: set[str] = set()

    # --- Hook: note each newcomer on channel attach ---
    @kit.hook(
        HookTrigger.ON_CHANNEL_ATTACHED,
        execution=HookExecution.ASYNC,
        name="note_newcomer",
    )
    async def note_newcomer(event: RoomEvent, ctx: RoomContext) -> None:
        if isinstance(event.content, SystemContent) and event.content.code == "channel_attached":
            channel_id = event.content.data.get("channel_id", "unknown")
            newcomers.add(channel_id)
            print(f"  [hook] Channel '{channel_id}' attached — welcome on its first message")

    # --- Hook: inject the welcome with the newcomer's first message ---
    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="inject_welcome")
    async def inject_welcome(event: RoomEvent, ctx: RoomContext) -> HookResult:
        channel_id = event.source.channel_id
        if channel_id not in newcomers:
            return HookResult.allow()
        newcomers.discard(channel_id)
        print(f"  [hook] First message from '{channel_id}' — injecting its welcome")
        return HookResult(action="allow", injected_events=[_welcome(event.room_id, channel_id)])

    # --- Hook: log room creation ---
    @kit.hook(
        HookTrigger.ON_ROOM_CREATED,
        execution=HookExecution.ASYNC,
        name="room_created_logger",
    )
    async def room_created_logger(event: RoomEvent, ctx: RoomContext) -> None:
        if isinstance(event.content, SystemContent):
            print(f"  [hook] Room created: {event.content.data.get('room_id')}")

    # --- Create room and attach channels ---
    print("Creating room...")
    await kit.create_room(room_id="welcome-room")

    print("\nAttaching Alice...")
    await kit.attach_channel("welcome-room", "ws-alice")

    print("Attaching Bob...")
    await kit.attach_channel("welcome-room", "ws-bob")

    # --- Each participant sends a first message ---
    print("\nAlice sends a greeting...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-alice",
            sender_id="alice",
            content=TextContent(body="Hey everyone, I just joined!"),
        )
    )

    print("Bob answers...")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-bob",
            sender_id="bob",
            content=TextContent(body="Hi Alice!"),
        )
    )

    # --- Show results ---
    print(f"\nAlice's inbox ({len(alice_inbox)} messages):")
    for ev in alice_inbox:
        body = getattr(ev.content, "body", str(ev.content))
        print(f"  <- [{ev.source.channel_id}] {body}")

    print(f"\nBob's inbox ({len(bob_inbox)} messages):")
    for ev in bob_inbox:
        body = getattr(ev.content, "body", str(ev.content))
        print(f"  <- [{ev.source.channel_id}] {body}")

    # --- Show stored history ---
    events = await kit.store.list_events("welcome-room")
    print(f"\nRoom history ({len(events)} events):")
    for ev in events:
        body = getattr(ev.content, "body", str(ev.content))
        print(f"  [{ev.type.value:>18}] {body}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
