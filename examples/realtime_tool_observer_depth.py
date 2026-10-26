"""A speech-to-speech model, a hook that tells it something at each of its
answers, and the depth limit that ends the loop between them.

The hook tells the model "lookup called" each time it calls a tool, by a
post into the room, then straight through ``channel.inject_text()``; then a
hook on the model's own words tells it "heard you" each time it speaks, and
one on its delegations (``ON_REALTIME_DELEGATION``) tells it "go on" each
time it hands reasoning to a backend. The model hears it and, being helpful,
answers again. What the hook sends is at
least as deep as the model's answer it reacts to, and a text whose answer
would be past ``max_chain_depth`` enters the model's session silently: it
hears it and answers nothing, so the loop ends at the limit, as one between
two agents does.

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
    """Answers every text it is asked to answer, from its own receive loop:
    by calling ``lookup``, by saying something when *speaks*, or by handing
    the reasoning to its hosted backend when *delegates*. A silent text asks
    it nothing, and ends the example."""

    def __init__(self, *, speaks: bool = False, delegates: bool = False) -> None:
        super().__init__()
        self.speaks = speaks
        self.delegates = delegates
        self.answers = 0
        self.heard = 0
        self.rested = asyncio.Event()

    def answer(self, session: Any) -> None:
        self.answers += 1
        if self.speaks:
            said = self.simulate_transcription(session, f"answer {self.answers}", "assistant")
        elif self.delegates:
            said = self.simulate_delegation(session, f"delegation-{self.answers}", "hosted")
        else:
            said = self.simulate_tool_call(session, f"call-{self.answers}", "lookup", {})
        asyncio.get_running_loop().create_task(said, context=contextvars.Context())

    async def inject_text(self, session: Any, text: str, **kwargs: Any) -> Any:
        result = await super().inject_text(session, text, **kwargs)
        self.heard += 1
        if kwargs.get("silent"):
            logger.info("The model hears %r and answers nothing", text)
            self.rested.set()
        elif self.answers < 20:
            self.answer(session)
        return result


async def run(tells: str) -> None:
    """The loop: a hook on the model's tool calls telling it by a post
    (*tells* ``"post"``) or by an injection (``"inject"``), a hook on its
    words telling it by an injection (``"words"``), or one on its delegations
    (``"delegation"``)."""
    model = EagerModel(speaks=tells == "words", delegates=tells == "delegation")

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
        elif tells == "inject":
            await channel.inject_text(session, f"{event.name} called")

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, execution=HookExecution.ASYNC, name="listener")
    async def listener(event: Any, ctx: Any) -> None:
        if tells == "words" and event.role == "assistant":
            await channel.inject_text(session, "heard you")

    @kit.hook(HookTrigger.ON_REALTIME_DELEGATION, execution=HookExecution.ASYNC, name="relay")
    async def relay(event: Any, ctx: Any) -> None:
        if tells == "delegation":
            await channel.inject_text(session, "go on")

    try:
        model.answer(session)
        await asyncio.wait_for(model.rested.wait(), 5)
        logger.info("Told by %s: the model answered %d times", tells, model.answers)
        assert (model.answers, model.heard) == (3, 3), (model.answers, model.heard)
    finally:
        await kit.close()


async def main() -> None:
    for tells in ("post", "inject", "words", "delegation"):
        await run(tells)


if __name__ == "__main__":
    asyncio.run(main())
