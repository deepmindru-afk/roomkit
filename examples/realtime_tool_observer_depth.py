"""A speech-to-speech model's tool call, an observer that posts about it, and
the depth limit that ends the loop between them.

An ASYNC ``ON_TOOL_CALL`` hook tells the model "lookup called" each time it
calls a tool: first by a post into the room, then straight through
``channel.inject_text()``. The model hears it and, being helpful, calls the
tool again. What the hook sends is at least as deep as the model's answer
that issued the call, and a text whose answer would be past
``max_chain_depth`` enters the model's session silently: it hears it and
answers nothing, so the loop ends at the limit, as one between two agents
does.

Run with: uv run python examples/realtime_tool_observer_depth.py
Uses a mock model; no credentials, external calls or audio hardware required.
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any

from shared import setup_logging

from roomkit import HookExecution, HookTrigger, RealtimeVoiceChannel, RoomKit
from roomkit.voice.realtime import MockRealtimeProvider, MockRealtimeTransport

logger = setup_logging("realtime_tool_observer_depth")

LOOKUP = {"name": "lookup", "description": "Look something up", "parameters": {"type": "object"}}
MAX_CHAIN_DEPTH = 4


class EagerModel(MockRealtimeProvider):
    """Calls ``lookup`` after every text it is asked to answer, from its own
    receive loop; a silent text asks it nothing, and ends the example."""

    def __init__(self) -> None:
        super().__init__()
        self.lookups = 0
        self.heard = 0
        self.rested = asyncio.Event()

    def call_lookup(self, session: Any) -> None:
        self.lookups += 1
        call = self.simulate_tool_call(session, f"call-{self.lookups}", "lookup", {})
        asyncio.get_running_loop().create_task(call, context=contextvars.Context())

    async def inject_text(self, session: Any, text: str, **kwargs: Any) -> Any:
        result = await super().inject_text(session, text, **kwargs)
        self.heard += 1
        if kwargs.get("silent"):
            logger.info("The model hears %r and answers nothing", text)
            self.rested.set()
        elif self.lookups < 20:
            self.call_lookup(session)
        return result


async def run(tells: str) -> None:
    """The loop, the hook telling the model by a post (*tells* ``"post"``) or
    by an injection into its session (``"inject"``)."""
    model = EagerModel()

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        return "found"

    kit = RoomKit(max_chain_depth=MAX_CHAIN_DEPTH)
    channel = RealtimeVoiceChannel(
        "voice",
        provider=model,
        transport=MockRealtimeTransport(),
        tools=[LOOKUP],
        tool_handler=lookup,
    )
    kit.register_channel(channel)
    await kit.create_room(room_id="call")
    await kit.attach_channel("call", "voice")
    session = await channel.start_session("call", "caller", object())

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="progress")
    async def progress(event: Any, ctx: Any) -> None:
        if tells == "post":
            await kit.post("call", f"{event.name} called")
        else:
            await channel.inject_text(session, f"{event.name} called")

    try:
        model.call_lookup(session)
        await asyncio.wait_for(model.rested.wait(), 5)
        logger.info("Told by %s: the model called lookup %d times", tells, model.lookups)
        assert (model.lookups, model.heard) == (3, 3), (model.lookups, model.heard)
    finally:
        await kit.close()


async def main() -> None:
    for tells in ("post", "inject"):
        await run(tells)


if __name__ == "__main__":
    asyncio.run(main())
