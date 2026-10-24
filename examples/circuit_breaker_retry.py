"""Retry and the circuit breaker on a channel's deliveries.

RoomKit retries a failed delivery and isolates a failing provider itself: the
binding's ``RetryPolicy`` sets the retries, and the router keeps one circuit
breaker per transport channel. Shows:

- a provider that fails twice: the message is delivered on the third attempt
  of the binding's ``RetryPolicy``;
- a provider that is down: once five deliveries failed, retries included, the
  channel's breaker opens, RoomKit publishes the ``circuit_breaker_opened``
  framework event, and the next deliveries are skipped without calling the
  provider. After 60 s the breaker lets one probe delivery through; a success
  closes it and publishes ``circuit_breaker_closed``.

Run with:
    uv run python examples/circuit_breaker_retry.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    FrameworkEvent,
    InboundMessage,
    ProviderResult,
    RoomEvent,
    RoomKit,
    SMSChannel,
    TextContent,
    WebSocketChannel,
)
from roomkit.models import RetryPolicy
from roomkit.providers.ai import ProviderError
from roomkit.providers.sms import MockSMSProvider

logger = setup_logging("example.circuit_breaker_retry")


class FlakySMSProvider(MockSMSProvider):
    """An SMS provider whose first ``failures`` sends fail."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures
        self.calls = 0

    async def send(self, event: RoomEvent, to: str, from_: str | None = None) -> ProviderResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderError(
                f"unreachable (call {self.calls})",
                provider="sms",
                status_code=503,
                retryable=True,
            )
        return await super().send(event, to, from_)


async def room_with(provider: FlakySMSProvider, retry_policy: RetryPolicy) -> RoomKit:
    """A room where what a user writes is sent out by SMS through ``provider``."""
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("ws-user"))
    kit.register_channel(SMSChannel("sms-out", provider=provider))
    await kit.create_room(room_id="alerts")
    await kit.attach_channel("alerts", "ws-user")
    await kit.attach_channel(
        "alerts",
        "sms-out",
        metadata={"phone_number": "+15550100"},
        retry_policy=retry_policy,
    )
    return kit


async def write(kit: RoomKit, body: str) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="ws-user", sender_id="user", content=TextContent(body=body))
    )


async def main() -> None:
    # Part 1: the binding's RetryPolicy rides out a provider that fails twice.
    logger.info("=== Retry ===")
    provider = FlakySMSProvider(failures=2)
    kit = await room_with(provider, RetryPolicy(max_retries=3, base_delay_seconds=0.05))
    await write(kit, "Server load above 90%")
    logger.info("provider calls: %d, messages sent: %d", provider.calls, len(provider.sent))
    await kit.close()

    # Part 2: a provider that is down trips the channel's circuit breaker.
    logger.info("=== Circuit breaker ===")
    provider = FlakySMSProvider(failures=1_000)
    kit = await room_with(provider, RetryPolicy(max_retries=1, base_delay_seconds=0.05))
    opened = asyncio.Event()

    @kit.on("circuit_breaker_opened")
    async def on_opened(event: FrameworkEvent) -> None:
        logger.info("breaker opened for %s", event.data["channel_id"])
        opened.set()

    @kit.on("circuit_breaker_closed")
    async def on_closed(event: FrameworkEvent) -> None:
        logger.info("breaker closed for %s", event.data["channel_id"])

    for i in range(5):
        await write(kit, f"Alert {i + 1}")
        logger.info("alert %d: provider calls so far %d", i + 1, provider.calls)
    await asyncio.wait_for(opened.wait(), timeout=5)

    calls = provider.calls
    for i in range(5, 7):
        await write(kit, f"Alert {i + 1}")
    logger.info(
        "alerts 6 and 7 were skipped: provider calls still %d (was %d)", provider.calls, calls
    )
    logger.info("one probe delivery goes through 60 s after the breaker opened")
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
