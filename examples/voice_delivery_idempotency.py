"""Deduplicate proactive voice delivery, including retries and a SQLite restart.

Run: uv run python examples/voice_delivery_idempotency.py
No credentials, network service or audio device required.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from roomkit import (
    DeliveryOutcome,
    RealtimeVoiceChannel,
    RoomKit,
    SQLiteStore,
    VoiceInjectionResult,
)
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

logger = logging.getLogger(__name__)


class DemoProvider(MockRealtimeProvider):
    """Simulate both a safe refusal and a lost confirmation after submission."""

    def __init__(self) -> None:
        super().__init__()
        self.refuse_once = True

    async def inject_text(
        self, session: VoiceSession, text: str, *, role: str = "user", silent: bool = False
    ) -> VoiceInjectionResult:
        if text == "Try again safely" and self.refuse_once:
            self.refuse_once = False
            return VoiceInjectionResult(status="not_sent", reason="not_connected", retryable=True)
        result = await super().inject_text(session, text, role=role, silent=silent)
        if text == "Confirmation lost":
            raise ConnectionError("Submission happened, but its confirmation was lost")
        return result


def show(label: str, result: DeliveryOutcome) -> None:
    logger.info(
        "%s: status=%s duplicate=%s reason=%s",
        label,
        result.status,
        result.duplicate,
        result.reason,
    )


async def demo(database: Path) -> None:
    provider = DemoProvider()
    channel = RealtimeVoiceChannel("voice", provider=provider, transport=MockRealtimeTransport())
    async with RoomKit(store=SQLiteStore(database)) as kit:
        kit.register_channel(channel)
        await kit.create_room(room_id="call")
        await kit.attach_channel("call", "voice")
        session = await channel.start_session("call", "alice", object())

        async def deliver(text: str, key: str) -> DeliveryOutcome:
            return await kit.deliver(
                "call", text, channel_id="voice", session_id=session.id, idempotency_key=key
            )

        # Two callers, one provider submission, one shared receipt.
        results = await asyncio.gather(
            deliver("Your result is ready", "event:1"), deliver("Your result is ready", "event:1")
        )
        for result in results:
            show("Concurrent delivery", result)
        assert all(result.status == "sent" for result in results)
        assert sum(result.duplicate for result in results) == 1
        assert len(provider.injected_texts) == 1

        # An exception after submission cannot authorize another injection.
        uncertain = await deliver("Confirmation lost", "event:2")
        replay = await deliver("Confirmation lost", "event:2")
        show("Lost confirmation", uncertain)
        show("Uncertain replay", replay)
        assert uncertain.status == replay.status == "unknown"
        assert replay.duplicate
        assert len(provider.injected_texts) == 2

        # Only a confirmed absence of submission permits a safe retry.
        refused = await deliver("Try again safely", "event:3")
        retried = await deliver("Try again safely", "event:3")
        show("Safe refusal", refused)
        show("Safe retry", retried)
        assert refused.error is not None
        assert refused.error.retryable
        assert retried.status == "sent"
        assert len(provider.injected_texts) == 3

    # The original voice session is gone. SQLite still knows what happened.
    async with RoomKit(store=SQLiteStore(database)) as restarted:
        for key, text, expected in (
            ("event:1", "Your result is ready", "sent"),
            ("event:2", "Confirmation lost", "unknown"),
        ):
            result = await restarted.deliver(
                "call", text, channel_id="voice", session_id=session.id, idempotency_key=key
            )
            show("Replay after restart", result)
            assert result.status == expected
            assert result.duplicate
    assert len(provider.injected_texts) == 3


async def main() -> None:
    with TemporaryDirectory(prefix="roomkit-voice-delivery-") as directory:
        await demo(Path(directory) / "rooms.db")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
