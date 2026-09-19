"""Count a room's events exactly, with the filter a page would use.

Demonstrates ``ConversationStore.get_event_count`` with an ``EventFilter``
(RFC §14.1): the count of a subset of the room, computed by the store with no
page and no cap, under exactly the conditions a ``list_events`` page would
apply. Shows:
- A ``BEFORE_BROADCAST`` hook refusing one message (stored ``BLOCKED``)
- The raw count: every committed row, the refused one included
- The filtered count: the messages the room received, what a page serves
- ``include_blocked`` lifting the received-rows default
- The count agreeing with the page under the same filter

Run with:
    uv run python examples/store_filtered_count.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from shared import setup_logging

from roomkit import (
    EventFilter,
    EventType,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)

ROOM_ID = "support-1"
CHANNEL_ID = "ws-user"
MESSAGES = [
    "Hello, my invoice for March is wrong",
    "I was charged twice for the same subscription",
    "SPAM: buy followers now",
    "Can I get a refund for the duplicate charge?",
    "Thanks, that solves it",
]


async def main() -> None:
    setup_logging("store_filtered_count")
    kit = RoomKit()
    kit.register_channel(WebSocketChannel(CHANNEL_ID))

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def refuse_spam(event: Any, _ctx: Any) -> HookResult:
        body = getattr(event.content, "body", "") or ""
        return HookResult.block("spam") if body.startswith("SPAM") else HookResult.allow()

    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, CHANNEL_ID)
    for text in MESSAGES:
        await kit.process_inbound(
            InboundMessage(channel_id=CHANNEL_ID, sender_id="user", content=TextContent(body=text))
        )

    store = kit.store
    messages = EventFilter(event_types=[EventType.MESSAGE])
    everything = await store.get_event_count(ROOM_ID)
    received = await store.get_event_count(ROOM_ID, messages)
    with_refused = await store.get_event_count(
        ROOM_ID, EventFilter(event_types=[EventType.MESSAGE], include_blocked=True)
    )
    page = await store.list_events(ROOM_ID, event_filter=messages)

    print(f"Every committed row, the refused one included: {everything}")
    print(f"Messages the room received (what a page serves): {received}")
    print(f"Messages, the refused one included:              {with_refused}")
    print(f"The page under the same filter holds {len(page)} rows: {[e.index for e in page]}")
    assert received == len(page)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
