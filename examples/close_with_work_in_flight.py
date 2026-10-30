"""Close a kit while work is in flight, and read what each call got back.

``kit.close()`` refuses every new call at once, cuts what has not started
and every model turn at once, gives the rest of the running work one grace
of five seconds, then cuts what is left. Each call it cut returns one result,
``cancellation_reason == "kit_closed"``, with what it delivered.

This example closes a kit that holds, all at once:

- a message whose send to ``slow-sms`` never ends (cut after the grace,
  reported failed, code ``kit_closed``, retryable), while ``web`` got it;
- a message queued behind it in the same room (dropped, never sent);
- an agent streaming a long answer in another room (cut at once, the text it
  streamed stored marked cancelled);
- a message whose BEFORE_BROADCAST hook runs one second (it ends in the
  grace: stored, its delivery dropped), and one waiting for that room's lock
  (cut at once, nothing stored);
- a call made once the close started (refused with ``KitClosedError``).

It takes about five seconds: the grace. Run with:
    uv run python examples/close_with_work_in_flight.py
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from roomkit import (
    Agent,
    ChannelCategory,
    ChannelOutput,
    HookResult,
    HookTrigger,
    InboundMessage,
    KitClosedError,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai import AIContext, MockAIProvider, StreamDone, StreamTextDelta

logger = logging.getLogger("examples.close_with_work_in_flight")


class StuckSMS(WebSocketChannel):
    """A transport whose provider never answers: every send hangs."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.sending = asyncio.Event()

    async def deliver(self, event: RoomEvent, binding: Any, context: Any) -> ChannelOutput:
        self.sending.set()
        await asyncio.Future()
        return ChannelOutput.empty()


class LongAnswer(MockAIProvider):
    """A model streaming a long answer, a word every 50 ms."""

    def __init__(self, started: asyncio.Event) -> None:
        super().__init__(streaming=True)
        self.started = started

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[Any]:
        yield StreamTextDelta(text="Here is ")
        for word in ("a", "long", "answer", "that", "goes", "on") * 100:
            await asyncio.sleep(0.05)
            self.started.set()
            yield StreamTextDelta(text=f"{word} ")
        yield StreamDone(finish_reason="stop", usage={"input_tokens": 1, "output_tokens": 600})


def message(body: str, channel: str = "chat") -> InboundMessage:
    return InboundMessage(channel_id=channel, sender_id="user", content=TextContent(body=body))


def summary(result: Any) -> str:
    deliveries = {
        channel: (r.status, r.error.code if r.error else None)
        for channel, r in result.delivery_results.items()
    }
    return (
        f"cancellation_reason={result.cancellation_reason!r} blocked={result.blocked} "
        f"reason={result.reason!r} delivery_results={deliveries}"
    )


async def main() -> None:
    started, in_hook = asyncio.Event(), asyncio.Event()
    stuck = StuckSMS("slow-sms")
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("chat"))
    kit.register_channel(WebSocketChannel("web"))
    kit.register_channel(stuck)
    kit.register_channel(Agent("agent", provider=LongAnswer(started)))
    for room_id in ("sends", "agent-room", "hooked"):
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "chat")
    await kit.attach_channel("sends", "web")
    await kit.attach_channel("sends", "slow-sms")
    await kit.attach_channel("agent-room", "agent", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="slow-check")
    async def slow_check(event: RoomEvent, ctx: Any) -> HookResult:
        if getattr(event.content, "body", "") == "checked slowly":
            in_hook.set()
            await asyncio.sleep(1)
        return HookResult.allow()

    # A send that never ends, and a message queued behind it.
    cut_send = await kit.process_inbound(message("hello"), room_id="sends", defer_delivery=True)
    await stuck.sending.wait()
    queued = await kit.process_inbound(message("queued"), room_id="sends", defer_delivery=True)
    # An agent streaming, a hook running, a call waiting for its room's lock.
    answer = asyncio.ensure_future(kit.process_inbound(message("Tell me"), room_id="agent-room"))
    await started.wait()
    hooked = asyncio.ensure_future(
        kit.process_inbound(message("checked slowly"), room_id="hooked")
    )
    await in_hook.wait()
    waiting = asyncio.ensure_future(kit.process_inbound(message("waiting"), room_id="hooked"))
    await asyncio.sleep(0.01)

    loop = asyncio.get_running_loop()
    began = loop.time()
    closing = asyncio.ensure_future(kit.close())
    await asyncio.sleep(0)
    try:
        await kit.send_event("hooked", "chat", TextContent(body="too late"))
    except KitClosedError as refused:
        logger.info("A call once the close started: %s", refused)
    await closing
    logger.info("close() returned after %.1f s", loop.time() - began)

    assert cut_send.delivery is not None
    assert queued.delivery is not None
    logger.info("The send that never ends: %s", summary(await cut_send.delivery.wait()))
    logger.info("The message queued behind it: %s", summary(await queued.delivery.wait()))
    result = await answer
    turn = dict(result.response_metadata).get("turns", {}).get("agent", {})
    logger.info(
        "The streaming turn: %s, its turn ended %r", summary(result), turn.get("loop_end_reason")
    )
    logger.info("The message whose hook ran 1 s: %s", summary(await hooked))
    logger.info("The message waiting for its lock: %s", summary(await waiting))
    for room_id in ("sends", "agent-room", "hooked"):
        stored = [
            f"{getattr(e.content, 'body', '')!r}{' (cancelled)' if e.metadata.get('cancelled') else ''}"
            for e in await kit.store.list_events(room_id)
            if e.type == "message"
        ]
        logger.info("Stored in %s: %s", room_id, ", ".join(stored))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("roomkit").setLevel(logging.WARNING)
    asyncio.run(main())
