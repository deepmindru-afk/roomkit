"""Rate limiting with TokenBucketRateLimiter.

Demonstrates how to configure and use rate limiting on channel bindings
to throttle outbound message delivery, and framework-level inbound rate
limiting to protect against message floods. Shows:
- RateLimit configuration (per-second, per-minute, per-hour)
- TokenBucketRateLimiter acquire() and wait() methods
- How rate limits apply to channel bindings
- Inbound rate limiting on RoomKit (drops excess messages before processing)

Run with:
    uv run python examples/rate_limiting.py
"""

from __future__ import annotations

import asyncio
import time

from roomkit import InboundMessage, RoomEvent, RoomKit, TextContent, WebSocketChannel
from roomkit.core.rate_limiter import TokenBucketRateLimiter
from roomkit.models.channel import RateLimit


async def main() -> None:
    # =====================================================
    # Part 1: TokenBucketRateLimiter directly
    # =====================================================
    print("=== TokenBucketRateLimiter Demo ===\n")

    limiter = TokenBucketRateLimiter()

    # Configure: 5 messages per second
    rate_limit = RateLimit(max_per_second=5.0)

    print("Sending 10 messages with 5/sec limit (acquire):")
    allowed = 0
    rejected = 0
    for i in range(10):
        if limiter.acquire("sms-channel", rate_limit):
            allowed += 1
            print(f"  Message {i + 1}: ALLOWED")
        else:
            rejected += 1
            print(f"  Message {i + 1}: RATE LIMITED")

    print(f"\n  Result: {allowed} allowed, {rejected} rate-limited")

    # =====================================================
    # Part 2: Using wait() to queue instead of drop
    # =====================================================
    print("\n\n=== Wait Mode (queue instead of drop) ===\n")

    limiter2 = TokenBucketRateLimiter()
    rate_limit_slow = RateLimit(max_per_second=3.0)

    print("Sending 6 messages with 3/sec limit (wait mode):")
    start = time.monotonic()

    for i in range(6):
        await limiter2.wait("slow-channel", rate_limit_slow)
        elapsed = time.monotonic() - start
        print(f"  Message {i + 1}: sent at {elapsed:.2f}s")

    total = time.monotonic() - start
    print(f"\n  Total time: {total:.2f}s (messages were queued, not dropped)")

    # =====================================================
    # Part 3: Rate limits on channel bindings
    # =====================================================
    print("\n\n=== Rate Limits on Channel Bindings ===\n")

    kit = RoomKit()

    ws_user = WebSocketChannel("ws-user")
    ws_sms = WebSocketChannel("ws-sms-sim")  # Simulates an SMS channel
    kit.register_channel(ws_user)
    kit.register_channel(ws_sms)

    # When each message reaches the rate-limited channel, from the first send.
    sms_received: list[tuple[str, float]] = []
    send_start = 0.0

    async def sms_recv(_conn: str, event: RoomEvent) -> None:
        body = event.content.body if isinstance(event.content, TextContent) else "?"
        sms_received.append((body, time.monotonic() - send_start))

    ws_user.register_connection("user-conn", lambda _c, _e: asyncio.sleep(0), room_id="rate-room")
    ws_sms.register_connection("sms-conn", sms_recv, room_id="rate-room")

    await kit.create_room(room_id="rate-room")
    await kit.attach_channel("rate-room", "ws-user")

    # Attach SMS channel with rate limit
    await kit.attach_channel(
        "rate-room",
        "ws-sms-sim",
        rate_limit=RateLimit(max_per_second=2.0, max_per_minute=60.0),
    )

    # Verify rate limit is stored on the binding
    binding = await kit.get_binding("rate-room", "ws-sms-sim")
    print("SMS channel rate limit:")
    print(f"  max_per_second: {binding.rate_limit.max_per_second}")  # type: ignore[union-attr]
    print(f"  max_per_minute: {binding.rate_limit.max_per_minute}")  # type: ignore[union-attr]

    # Send a burst: none is dropped, delivery to the SMS channel is paced
    # (the bucket holds 2, then refills at 2/sec) and process_inbound waits
    # for it.
    print("\nSending 5 messages at once through the rate-limited channel:")
    send_start = time.monotonic()
    for i in range(5):
        result = await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user",
                sender_id="user",
                content=TextContent(body=f"Message {i + 1}"),
            )
        )
        sent_at = time.monotonic() - send_start
        print(f"  Message {i + 1}: returned at {sent_at:.2f}s, blocked={result.blocked}")
    print("\nDelivered to the SMS channel (2/sec):")
    for body, at in sms_received:
        print(f"  {body}: delivered at {at:.2f}s")

    # =====================================================
    # Part 4: Different rate configurations
    # =====================================================
    print("\n\n=== Rate Limit Configurations ===\n")

    configs = [
        ("High throughput", RateLimit(max_per_second=100.0)),
        ("Standard SMS", RateLimit(max_per_second=1.0, max_per_minute=30.0)),
        ("Hourly newsletter", RateLimit(max_per_hour=100.0)),
        (
            "Burst control",
            RateLimit(
                max_per_second=10.0,
                max_per_minute=200.0,
                max_per_hour=5000.0,
            ),
        ),
    ]

    for name, config in configs:
        rates = []
        if config.max_per_second:
            rates.append(f"{config.max_per_second}/sec")
        if config.max_per_minute:
            rates.append(f"{config.max_per_minute}/min")
        if config.max_per_hour:
            rates.append(f"{config.max_per_hour}/hr")
        print(f"  {name:20s}: {', '.join(rates)}")

    await kit.close()

    # =====================================================
    # Part 5: Inbound rate limiting (framework-level)
    # =====================================================
    print("\n\n=== Inbound Rate Limiting ===\n")

    # Protect the inbound pipeline — drop excess messages before any processing
    kit2 = RoomKit(inbound_rate_limit=RateLimit(max_per_second=3.0))

    ws_in = WebSocketChannel("ws-inbound")
    kit2.register_channel(ws_in)
    ws_in.register_connection("conn", lambda _c, _e: asyncio.sleep(0), room_id="rate-room-2")

    await kit2.create_room(room_id="rate-room-2")
    await kit2.attach_channel("rate-room-2", "ws-inbound")

    print("Sending 8 messages with inbound limit of 3/sec:")
    allowed_count = 0
    for i in range(8):
        result = await kit2.process_inbound(
            InboundMessage(
                channel_id="ws-inbound",
                sender_id="user",
                content=TextContent(body=f"Message {i + 1}"),
            )
        )
        status = "ALLOWED" if not result.blocked else f"BLOCKED ({result.reason})"
        print(f"  Message {i + 1}: {status}")
        if not result.blocked:
            allowed_count += 1

    print(f"\n  Result: {allowed_count} allowed, {8 - allowed_count} rate-limited")

    await kit2.close()


if __name__ == "__main__":
    asyncio.run(main())
