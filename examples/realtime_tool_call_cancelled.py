"""A tool call the model abandons: the handler stops, the audit sees it.

    uv run python examples/realtime_tool_call_cancelled.py

When the caller interrupts Gemini Live while a tool call is outstanding, the
server sends ``tool_call_cancellation``: it will not read the result. RoomKit
carries that to the application through ``on_tool_call_cancelled``, cancels
the handler still running for the call, sends nothing back, and reports the
call to ``ON_TOOL_CALL``'s async observers with ``cancelled=True`` beside
``is_error=True``. Before, the handler ran to the end for a result the
provider then dropped in silence, and no hook saw the call end.

Runs on the mock provider, so it needs no key and no microphone: the
cancellation is simulated where Gemini would emit it. A real Gemini Live
session walks the same path when the caller talks over the model while a
background tool runs (see ``examples/realtime_background_tools.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from shared import setup_logging

from roomkit import (
    HookExecution,
    HookResult,
    HookTrigger,
    RealtimeVoiceChannel,
    RoomContext,
    RoomKit,
    ToolCallEvent,
)
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

logger = setup_logging("roomkit.examples.realtime_tool_call_cancelled")

TOOL_SECONDS = 5.0


@dataclass
class Ledger:
    """What the audit hook and the handler saw, in order."""

    outcomes: list[tuple[str, str]]
    handler_started: asyncio.Event
    handler_interrupted_after: float | None = None


async def run() -> Ledger:
    provider = MockRealtimeProvider()
    transport = MockRealtimeTransport()
    ledger = Ledger(outcomes=[], handler_started=asyncio.Event())

    async def check_inventory(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """A slow lookup: the kind of work the model abandons on a barge-in."""
        started = time.monotonic()
        ledger.handler_started.set()
        try:
            await asyncio.sleep(TOOL_SECONDS)
        except asyncio.CancelledError:
            ledger.handler_interrupted_after = round(time.monotonic() - started, 2)
            logger.info("handler for %s interrupted: the model abandoned the call", name)
            raise
        return {"item": arguments.get("item"), "in_stock": 42}

    channel = RealtimeVoiceChannel(
        "warehouse",
        provider=provider,
        transport=transport,
        tools=[
            {
                "name": "check_inventory",
                "description": "Look up warehouse stock for one item.",
                "parameters": {
                    "type": "object",
                    "properties": {"item": {"type": "string"}},
                    "required": ["item"],
                },
            }
        ],
        tool_handler=check_inventory,
    )
    kit = RoomKit()
    kit.register_channel(channel)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="tool_auditor")
    async def tool_auditor(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        # The outcome is stated on the event: a cancelled call is not a
        # refusal, and a refusal is not a served call.
        outcome = "cancelled" if event.cancelled else "refused" if event.is_error else "ok"
        ledger.outcomes.append((event.tool_call_id, outcome))
        logger.info("audit: %s(%s) %s", event.name, event.tool_call_id, outcome)
        return HookResult.allow()

    room = await kit.create_room()
    await kit.attach_channel(room.id, channel.channel_id)
    session = await channel.start_session(room.id, "caller", object())
    try:
        # The model asks for a lookup, then the caller talks over it: Gemini
        # sends tool_call_cancellation for the outstanding call.
        await provider.simulate_tool_call(session, "call-1", "check_inventory", {"item": "widget"})
        await asyncio.wait_for(ledger.handler_started.wait(), timeout=2)
        await asyncio.sleep(0.2)
        await provider.simulate_tool_call_cancellation(session, ["call-1"])
        await asyncio.sleep(0.2)

        # A second call, left alone, is served as usual.
        await provider.simulate_tool_call(session, "call-2", "check_inventory", {"item": "bolt"})
        await asyncio.sleep(TOOL_SECONDS + 0.5)
    finally:
        await kit.close()

    logger.info(
        "results sent to the provider: %s (call-1 never, call-2 once)",
        [call_id for _sid, call_id, _body in provider.tool_results],
    )
    return ledger


async def main() -> None:
    ledger = await run()
    logger.info(
        "handler interrupted after %ss of a %ss lookup; ledger: %s",
        ledger.handler_interrupted_after,
        TOOL_SECONDS,
        ledger.outcomes,
    )


if __name__ == "__main__":
    asyncio.run(main())
