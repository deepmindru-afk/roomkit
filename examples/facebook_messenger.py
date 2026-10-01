"""Facebook Messenger example — receive and send messages via Messenger.

Offline demo: a sample Messenger webhook is parsed into the room, then an
agent's reply is delivered back to the user through the Messenger channel.
``MockMessengerProvider`` records that send instead of calling the Graph API,
so no token is needed and nothing is sent. For a real page, swap the provider::

    from roomkit.providers.messenger import FacebookMessengerProvider, MessengerConfig

    provider = FacebookMessengerProvider(MessengerConfig(page_access_token=...))

Run with:
    uv run python examples/facebook_messenger.py
"""

from __future__ import annotations

import asyncio

from roomkit import InboundMessage, RoomKit, TextContent, WebSocketChannel
from roomkit.channels import MessengerChannel
from roomkit.providers.messenger import MockMessengerProvider, parse_messenger_webhook


async def main() -> None:
    # --- Provider ------------------------------------------------------------
    provider = MockMessengerProvider()

    # --- RoomKit setup -------------------------------------------------------
    kit = RoomKit()
    messenger = MessengerChannel("msg-main", provider=provider)
    ws = WebSocketChannel("ws-agent")
    kit.register_channel(messenger)
    kit.register_channel(ws)

    await kit.create_room(room_id="demo-room")
    # The recipient of what the room sends on Messenger: the user's PSID.
    await kit.attach_channel(
        "demo-room",
        "msg-main",
        metadata={"facebook_user_id": "USER_PSID"},
    )
    await kit.attach_channel("demo-room", "ws-agent")

    # --- Simulate inbound webhook --------------------------------------------
    raw_webhook = {
        "object": "page",
        "entry": [
            {
                "id": "PAGE_ID",
                "time": 1700000000000,
                "messaging": [
                    {
                        "sender": {"id": "USER_PSID"},
                        "recipient": {"id": "PAGE_ID"},
                        "timestamp": 1700000000000,
                        "message": {
                            "mid": "mid.example",
                            "text": "Hello from Messenger!",
                        },
                    }
                ],
            }
        ],
    }

    inbound_messages = parse_messenger_webhook(raw_webhook, channel_id="msg-main")
    for inbound in inbound_messages:
        print(f"Parsed inbound from {inbound.sender_id}: {inbound.content.body}")  # type: ignore[union-attr]
        result = await kit.process_inbound(inbound, room_id="demo-room")
        print(f"  Processed: blocked={result.blocked}")

    # --- Agent replies: the room delivers it to Messenger --------------------
    reply = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-agent",
            sender_id="agent",
            content=TextContent(body="Hi! How can I help?"),
        ),
        room_id="demo-room",
    )
    outcome = reply.delivery_results.get("msg-main")
    print(f"\nReply delivered on msg-main: {outcome.status if outcome else 'not delivered'}")
    for sent in provider.sent:
        print(f"  Messenger send to={sent['to']}: {sent['event'].content.body}")

    # --- Show conversation history -------------------------------------------
    events = await kit.store.list_events("demo-room")
    print(f"\nRoom history ({len(events)} events):")
    for ev in events:
        print(f"  [{ev.source.channel_id}] {ev.content.body}")  # type: ignore[union-attr]

    await kit.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
