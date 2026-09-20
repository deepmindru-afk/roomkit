"""Billing a realtime call: what the provider reports, and where to read it.

    uv run python examples/realtime_usage_accounting.py

A spoken turn's cost is known to the service that billed it, so the provider
reports it and RoomKit relays it unaltered (RFC §12.4.2). Two surfaces carry
that report:

- ``provider.on_usage(callback)`` fires on every report, with the session and
  the map just recorded. This is the one to bill from.
- ``session.last_usage`` is the snapshot the last report left behind — handy
  to read at the end of a turn, and only that: the next report replaces it and
  the channel clears it after each turn, so a ledger built by polling it comes
  out short. This example shows both, side by side, and the gap between them.

The two totals, ``input_tokens`` and ``output_tokens``, are the only keys the
framework fixes. Everything beside them is the provider's own: Gemini Live
sends its per-modality counts and cached share, OpenAI Realtime its
``input_token_details``, GPT-Live seconds rather than tokens. An absent key
means unreported, never zero — which is why this ledger sums what it finds
instead of assuming a shape.

Runs on the mock provider, so it needs no key and no microphone. A real
session swaps two lines::

    from roomkit.providers.gemini import GeminiLiveProvider

    provider = GeminiLiveProvider(api_key=...)
    provider.on_usage(ledger.record)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
from dataclasses import dataclass, field
from typing import Any

from shared import setup_logging

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

logger = setup_logging("roomkit.examples.realtime_usage_accounting")

# What a spoken turn reports on Gemini Live: two totals, and the breakdown
# that says what they are made of. Audio and text are priced apart, and the
# cached share is priced apart again.
TURNS = [
    (23520, 44, {"prompt_tokens_details": {"AUDIO": 3520, "TEXT": 20000}}),
    (
        41200,
        61,
        {
            "prompt_tokens_details": {"AUDIO": 5200, "TEXT": 36000},
            "cached_content_token_count": 31000,
        },
    ),
    (58900, 38, {"cached_content_token_count": 47000, "thoughts_token_count": 300}),
]


@dataclass
class Ledger:
    """What a host billing the call keeps, one row per report."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def record(self, session: VoiceSession, usage: dict[str, Any]) -> None:
        """Registered through ``on_usage``: called on every report, sync or async."""
        cached = int(usage.get("cached_content_token_count") or 0)
        fresh = int(usage.get("input_tokens") or 0) - cached
        self.rows.append(
            {
                "session": session.id,
                "fresh_input": fresh,
                "cached_input": cached,
                "output": int(usage.get("output_tokens") or 0),
                "audio_input": int(usage.get("prompt_tokens_details", {}).get("AUDIO") or 0),
            }
        )
        logger.info("usage reported: %s", self.rows[-1])

    @property
    def cached_share(self) -> float:
        billed = sum(r["fresh_input"] + r["cached_input"] for r in self.rows)
        return sum(r["cached_input"] for r in self.rows) / billed if billed else 0.0


async def run() -> tuple[Ledger, list[int], dict[str, Any]]:
    provider = MockRealtimeProvider()
    transport = MockRealtimeTransport()
    ledger = Ledger()
    provider.on_usage(ledger.record)

    channel = RealtimeVoiceChannel("support-line", provider=provider, transport=transport)
    kit = RoomKit()
    kit.register_channel(channel)

    room = await kit.create_room()
    await kit.attach_channel(room.id, channel.channel_id)
    session = await channel.start_session(room.id, "caller", object())

    # A host that polls the snapshot instead of listening sees one turn at a
    # time — and only the turns it happens to look at.
    polled: list[int] = []
    try:
        for input_tokens, output_tokens, details in TURNS:
            await provider.simulate_usage(session, input_tokens, output_tokens, details=details)
            polled.append(session.last_usage["input_tokens"])
        snapshot = session.last_usage
    finally:
        await kit.close()

    return ledger, polled, snapshot


async def main() -> None:
    ledger, polled, snapshot = await run()

    logger.info("the callback billed %d turns: %s", len(ledger.rows), ledger.rows)
    logger.info("cache served %.0f%% of the input billed", ledger.cached_share * 100)
    logger.info("the snapshot kept the last turn only: %s", snapshot)
    logger.info(
        "a poll after each turn saw %s — right here, wrong the moment a turn lands "
        "between two reads, and empty once the channel clears it",
        polled,
    )


if __name__ == "__main__":
    asyncio.run(main())
