"""A caller that stops waiting (RFC §13.6).

Once its call is decided, a caller cancelled (a client gone, an outer timeout
of its own) leaves the call's writes whole: the event a commit in flight
wrote is delivered, what its hooks injected is committed, and the room's
lane goes on.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.store.memory import InMemoryStore
from tests.test_framework import SimpleChannel


def _body(event: RoomEvent) -> str:
    return getattr(event.content, "body", "")


def _msg(body: str, sender: str = "u1") -> InboundMessage:
    return InboundMessage(channel_id="sms1", sender_id=sender, content=TextContent(body=body))


class _SlowAnswer(InMemoryStore):
    """A commit whose answer travels back: the row is written and the call
    has not returned yet, as over a database round trip."""

    def __init__(self) -> None:
        super().__init__()
        self.held: str | None = None
        self.written = asyncio.Event()
        self.answer = asyncio.Event()

    async def commit_event(self, room_id: str, event: RoomEvent, **kw: Any) -> RoomEvent:
        committed = await super().commit_event(room_id, event, **kw)
        if self.held is not None and _body(event) == self.held:
            self.held = None
            self.written.set()
            await self.answer.wait()
        return committed


async def _cancelled_while(caller: asyncio.Task[Any], reached: asyncio.Event) -> None:
    """Cancel *caller* once it *reached* the point under test, and see the
    cancellation reach it."""
    await asyncio.wait_for(reached.wait(), 1)
    caller.cancel()


class TestACommitInFlight:
    @pytest.mark.parametrize("action", ["allow", "block"])
    async def test_its_writes_are_whole_and_the_lane_goes_on(self, action: str) -> None:
        """The caller is cancelled while its event's commit (a DELIVERED event
        or a BLOCKED record) is in flight: the event is delivered, what its
        hook injected is committed and delivered, and the room's next message
        waits for no gap (RFC §13.6, §9.4, §10.1 step 12)."""
        store = _SlowAnswer()
        kit = RoomKit(store=store, delivery_gap_timeout=0.2)
        ws = SimpleChannel("ws1")
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="notify")
        async def notify(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) != "first":
                return HookResult.allow()
            notice = RoomEvent(
                room_id="r1",
                source=EventSource(channel_id="sms1", channel_type="sms"),
                content=TextContent(body="notice"),
            )
            return HookResult(
                action=action,
                reason="held" if action == "block" else None,
                injected_events=[InjectedEvent(event=notice, target_channel_ids=["ws1"])],
            )

        store.held = "first"
        try:
            caller = asyncio.create_task(kit.process_inbound(_msg("first"), room_id="r1"))
            await _cancelled_while(caller, store.written)
            store.answer.set()
            with pytest.raises(asyncio.CancelledError):
                await caller
            await asyncio.wait_for(kit.process_inbound(_msg("second"), room_id="r1"), 1)

            first = ["first"] if action == "allow" else []
            assert [_body(e) for e in ws.delivered] == [*first, "notice", "second"]
        finally:
            await kit.close()


class TestTheKitsClose:
    async def test_waits_for_the_writes_a_cancelled_caller_left(self) -> None:
        """A host cancels a call whose commit is in flight and closes the kit
        at once: the close waits for the writes, never cuts them, and the
        event is delivered before the channels close (RFC §13.6)."""
        store = _SlowAnswer()
        kit = RoomKit(store=store)
        ws = SimpleChannel("ws1")
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")
        store.held = "first"

        caller = asyncio.create_task(kit.process_inbound(_msg("first"), room_id="r1"))
        await _cancelled_while(caller, store.written)
        closing = asyncio.ensure_future(kit.close())
        await asyncio.wait({closing}, timeout=0.05)
        assert not closing.done()  # it waits for the commit in flight
        store.answer.set()
        await asyncio.wait_for(closing, 1)

        assert [_body(e) for e in ws.delivered] == ["first"]
        with pytest.raises(asyncio.CancelledError):
            await caller
