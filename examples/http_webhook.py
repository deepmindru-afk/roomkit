"""HTTP webhook example — receive and send messages via generic HTTP.

Offline demo: a sample inbound HTTP payload is parsed into the room, then an
agent's reply is delivered back to the sender through the HTTP channel.
``MockHTTPProvider`` records that send instead of POSTing it, so no URL is
needed and nothing leaves the machine. For a real endpoint, swap the
provider::

    from roomkit.providers.http import HTTPProviderConfig, WebhookHTTPProvider

    provider = WebhookHTTPProvider(HTTPProviderConfig(webhook_url="https://example.com/hook"))

Run with:
    uv run python examples/http_webhook.py
"""

from __future__ import annotations

import asyncio

from roomkit import InboundMessage, RoomKit, TextContent, WebSocketChannel
from roomkit.channels import HTTPChannel
from roomkit.providers.http import MockHTTPProvider, parse_http_webhook


async def main() -> None:
    # --- Provider ------------------------------------------------------------
    provider = MockHTTPProvider()

    # --- RoomKit setup -------------------------------------------------------
    kit = RoomKit()
    http_channel = HTTPChannel("http-main", provider=provider)
    ws = WebSocketChannel("ws-agent")
    kit.register_channel(http_channel)
    kit.register_channel(ws)

    await kit.create_room(room_id="demo-room")
    # The recipient of what the room sends over HTTP.
    await kit.attach_channel(
        "demo-room",
        "http-main",
        metadata={"recipient_id": "user-123"},
    )
    await kit.attach_channel("demo-room", "ws-agent")

    # --- Simulate inbound webhook --------------------------------------------
    raw_payload = {
        "sender_id": "user-123",
        "body": "Hello from HTTP!",
        "external_id": "msg-001",
    }

    inbound = parse_http_webhook(raw_payload, channel_id="http-main")
    print(f"Parsed inbound from {inbound.sender_id}: {inbound.content.body}")  # type: ignore[union-attr]
    result = await kit.process_inbound(inbound, room_id="demo-room")
    print(f"  Processed: blocked={result.blocked}")

    # --- Agent replies: the room delivers it over HTTP -----------------------
    reply = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-agent",
            sender_id="agent",
            content=TextContent(body="Thanks, we received your message."),
        ),
        room_id="demo-room",
    )
    outcome = reply.delivery_results.get("http-main")
    print(f"\nReply delivered on http-main: {outcome.status if outcome else 'not delivered'}")
    for sent in provider.sent:
        print(f"  HTTP send to={sent['to']}: {sent['event'].content.body}")

    # --- Show conversation history -------------------------------------------
    events = await kit.store.list_events("demo-room")
    print(f"\nRoom history ({len(events)} events):")
    for ev in events:
        print(f"  [{ev.source.channel_id}] {ev.content.body}")  # type: ignore[union-attr]

    await kit.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
