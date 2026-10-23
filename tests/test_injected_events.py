"""The events a hook injects, each committed as a pass of its cause's
cascade (RFC §9.4, §8.3, §10.1 step 14)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

import pytest

from roomkit import RoomKit
from roomkit.core.locks import InMemoryLockManager, _has_room_lock
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelType,
    EventStatus,
    EventType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import EventContent, EventSource, RichContent, RoomEvent, TextContent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.identity import IdentityHookResult, IdentityResult
from roomkit.models.store_filter import EventFilter
from roomkit.models.task import Task
from tests.test_framework import AILikeChannel, SimpleChannel
from tests.test_identity_unknown_outcomes import UnknownResolver


def _notice(
    body: str = "notice",
    *,
    source: str = "system",
    content: EventContent | None = None,
    **fields: object,
) -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(channel_id=source, channel_type=ChannelType.SYSTEM),
        content=content or TextContent(body=body),
        **fields,  # type: ignore[arg-type]
    )


def _body(event: RoomEvent) -> str | None:
    return event.content.body if isinstance(event.content, TextContent) else None


async def _room(**kit_args: object) -> tuple[RoomKit, SimpleChannel, AILikeChannel]:
    """A room with an SMS sender, a WebSocket transport and an agent."""
    kit = RoomKit(**kit_args)  # type: ignore[arg-type]
    ws, ai = SimpleChannel("ws"), AILikeChannel("ai")
    await kit.create_room(room_id="r1")
    for channel in (SimpleChannel("sms"), ws, ai):
        kit.register_channel(channel)
    await kit.attach_channel("r1", "sms")
    await kit.attach_channel("r1", "ws")
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
    return kit, ws, ai


def _on_hello(
    kit: RoomKit, *injected: InjectedEvent, block: bool = False
) -> Callable[..., object]:
    """A hook that injects *injected* when the SMS sender says hello."""

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="inject_on_hello")
    async def inject(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if _body(event) != "hello":
            return HookResult.allow()
        if block:
            return HookResult.block("held", injected=list(injected))
        return HookResult.allow(injected=list(injected))

    return inject


async def _hello(kit: RoomKit) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="hello"))
    )


async def _stored(kit: RoomKit) -> list[RoomEvent]:
    return await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))


async def _notices(kit: RoomKit, body: str = "notice") -> list[RoomEvent]:
    return [e for e in await _stored(kit) if _body(e) == body]


class _WatchedLocks(InMemoryLockManager):
    """Room locks that tell how many callers asked for one."""

    def __init__(self) -> None:
        super().__init__()
        self.watch()

    def watch(self) -> None:
        """Count the callers asking for a lock from now on."""
        self.asked = 0
        self._reached: dict[int, asyncio.Event] = {}

    def reached(self, count: int) -> asyncio.Event:
        """Set once *count* callers have asked for a lock since :meth:`watch`."""
        event = self._reached.setdefault(count, asyncio.Event())
        if self.asked >= count:
            event.set()
        return event

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
        if not _has_room_lock(room_id, self):
            self.asked += 1
            for count, event in self._reached.items():
                if self.asked >= count:
                    event.set()
        async with super().locked(room_id):
            yield


class _HeldHello:
    """A room with no agent whose hook holds hello's section until ``go``,
    then injects a notice to ``ws``, blocking hello when *block*; ``noticed``
    is set once the notice's delivery set has run."""

    def __init__(self, *, block: bool) -> None:
        self.locks = _WatchedLocks()
        self.kit = RoomKit(lock_manager=self.locks)
        self.in_section, self.go, self.noticed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        notice = InjectedEvent(event=_notice(), target_channel_ids=["ws"])

        @self.kit.hook(HookTrigger.BEFORE_BROADCAST, name="slow_notifier")
        async def slow_notifier(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) != "hello":
                return HookResult.allow()
            self.in_section.set()
            await self.go.wait()
            if block:
                return HookResult.block("held", injected=[notice])
            return HookResult.allow(injected=[notice])

        @self.kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="seen")
        async def seen(event: RoomEvent, ctx: RoomContext) -> None:
            if _body(event) == "notice":
                self.noticed.set()

    async def open(self) -> RoomKit:
        await self.kit.create_room(room_id="r1")
        for channel in (SimpleChannel("sms"), SimpleChannel("ws")):
            self.kit.register_channel(channel)
            await self.kit.attach_channel("r1", channel.channel_id)
        self.locks.watch()
        return self.kit


class TestItsPass:
    @pytest.mark.parametrize("block", [False, True], ids=["cause delivered", "cause blocked"])
    async def test_commits_once_its_cause_released_the_lock(self, block: bool) -> None:
        """RFC §9.4, §10.1 steps 10 to 14: it commits in a section of its
        own once its cause's has released the room lock, so an event
        waiting for the lock meanwhile commits between them."""
        room = _HeldHello(block=block)
        kit = await room.open()
        try:
            cause = asyncio.create_task(_hello(kit))
            await room.in_section.wait()
            # A direct injection asks for the lock at step 6 alone.
            later = asyncio.create_task(kit.send_event("r1", "sms", TextContent(body="later")))
            await room.locks.reached(2).wait()
            room.go.set()
            await asyncio.gather(cause, later)
            stored = [_body(e) for e in await _stored(kit) if e.type == EventType.MESSAGE]
            assert stored == ["hello", "later", "notice"]
        finally:
            await kit.close()

    async def test_is_committed_when_its_caller_is_cancelled_waiting_for_the_lock(
        self,
    ) -> None:
        """RFC §9.4: committed whichever way its cause ended, the caller
        that disconnects while another holds the lock included."""
        room = _HeldHello(block=True)
        kit = await room.open()
        release = asyncio.Event()

        async def hold() -> None:
            async with room.locks.locked("r1"):
                await release.wait()

        try:
            cause = asyncio.create_task(_hello(kit))
            await room.in_section.wait()
            holder = asyncio.create_task(hold())
            await room.locks.reached(2).wait()
            room.go.set()
            # The notice's pass asks for the lock, behind the holder.
            await room.locks.reached(3).wait()
            cause.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cause
            release.set()
            await holder
            await asyncio.wait_for(room.noticed.wait(), 1)
            assert [e.status for e in await _notices(kit)] == [EventStatus.DELIVERED]
        finally:
            await kit.close()

    async def test_is_committed_when_a_later_step_of_its_caller_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        room = _HeldHello(block=True)
        kit = await room.open()
        room.go.set()

        def fails(*args: object) -> None:
            raise RuntimeError("after the decision")

        monkeypatch.setattr(kit, "_name_held", fails)
        try:
            with pytest.raises(RuntimeError, match="after the decision"):
                await _hello(kit)
            await asyncio.wait_for(room.noticed.wait(), 1)
            assert [e.status for e in await _notices(kit)] == [EventStatus.DELIVERED]
        finally:
            await kit.close()

    async def test_is_committed_after_its_cause(self) -> None:
        kit, _, _ = await _room()
        _on_hello(kit, InjectedEvent(event=_notice(), target_channel_ids=["ws"]))
        try:
            await _hello(kit)
            hello = next(e for e in await _stored(kit) if _body(e) == "hello")
            (notice,) = await _notices(kit)
            # Another event may commit between them (RFC §9.4): the agent's
            # answer to hello, whose pass may take the lock first.
            assert notice.index > hello.index
        finally:
            await kit.close()

    async def test_one_a_direct_injection_s_hooks_inject_waits_for_the_section_holding_the_lock(
        self,
    ) -> None:
        """RFC §9.4: an event code under the lock injects directly commits
        ahead of its cause, within the cause's section; what its own hooks
        inject joins that section's queue, committed once it released the
        lock, after the cause."""
        kit, _, _ = await _room()

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="direct")
        async def direct(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) == "hello":
                await kit.send_event("r1", "sms", TextContent(body="aside"))
            if _body(event) == "aside":
                notice = InjectedEvent(event=_notice(), target_channel_ids=["ws"])
                return HookResult.allow(injected=[notice])
            return HookResult.allow()

        try:
            await _hello(kit)
            ours = [
                _body(e) for e in await _stored(kit) if _body(e) in ("hello", "aside", "notice")
            ]
            assert ours == ["aside", "hello", "notice"]
        finally:
            await kit.close()

    async def test_is_committed_when_its_cause_is_blocked(self) -> None:
        kit, ws, _ = await _room()
        _on_hello(kit, InjectedEvent(event=_notice(), target_channel_ids=["ws"]), block=True)
        try:
            await _hello(kit)
            assert [_body(e) for e in ws.delivered] == ["notice"]
        finally:
            await kit.close()

    async def test_is_committed_when_its_source_may_not_write(self) -> None:
        kit, ws, _ = await _room()
        await kit.mute("r1", "sms")
        _on_hello(kit, InjectedEvent(event=_notice(), target_channel_ids=["ws"]))
        try:
            await _hello(kit)
            hello = next(e for e in await _stored(kit) if _body(e) == "hello")
            assert (hello.blocked_by, [_body(e) for e in ws.delivered]) == (
                "source_muted",
                ["notice"],
            )
        finally:
            await kit.close()

    async def test_runs_the_before_broadcast_hooks_that_may_block_it(self) -> None:
        kit, ws, _ = await _room()
        _on_hello(kit, InjectedEvent(event=_notice(), target_channel_ids=["ws"]))

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="no_notices")
        async def no_notices(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("quiet") if _body(event) == "notice" else HookResult.allow()

        try:
            await _hello(kit)
            (notice,) = await _notices(kit)
            assert (notice.status, notice.blocked_by) == (EventStatus.BLOCKED, "no_notices")
            assert "notice" not in [_body(e) for e in ws.delivered]
        finally:
            await kit.close()

    async def test_commits_what_its_own_hooks_inject_after_it(self) -> None:
        kit, _, _ = await _room()
        _on_hello(kit, InjectedEvent(event=_notice(), target_channel_ids=["ws"]))

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="echo")
        async def echo(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) != "notice":
                return HookResult.allow()
            echoed = InjectedEvent(event=_notice("echo"), target_channel_ids=["ws"])
            return HookResult.allow(injected=[echoed])

        try:
            await _hello(kit)
            (notice,) = await _notices(kit)
            (echoed,) = await _notices(kit, "echo")
            assert echoed.index > notice.index
        finally:
            await kit.close()

    async def test_counts_against_the_reentry_budget(self) -> None:
        kit, _, _ = await _room(max_chain_depth=1)

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="always_inject")
        async def always_inject(event: RoomEvent, ctx: RoomContext) -> HookResult:
            # Every event injects one more, its own injections included.
            return HookResult.allow(
                injected=[InjectedEvent(event=_notice(), target_channel_ids=["ws"])]
            )

        try:
            await _hello(kit)
            notices = await _notices(kit)
            assert [e.blocked_by for e in notices] == [None] * 10 + ["reentry_loop_cap"]
        finally:
            await kit.close()


class TestWhatItIsCommittedWith:
    async def test_keeps_its_cause_s_chain_depth(self) -> None:
        kit, _, _ = await _room()
        _on_hello(kit, InjectedEvent(event=_notice(chain_depth=4), target_channel_ids=["ws"]))
        try:
            await _hello(kit)
            assert [e.chain_depth for e in await _notices(kit)] == [0]
        finally:
            await kit.close()

    async def test_keeps_the_depth_of_an_answer_that_caused_it(self) -> None:
        kit, _, _ = await _room()

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="on_answer")
        async def on_answer(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) != "AI reply":
                return HookResult.allow()
            notice = InjectedEvent(event=_notice(), target_channel_ids=["ws"])
            return HookResult.allow(injected=[notice])

        try:
            await _hello(kit)
            assert [e.chain_depth for e in await _notices(kit)] == [1]
        finally:
            await kit.close()

    async def test_names_its_bound_targets_as_its_visibility(self) -> None:
        kit, _, _ = await _room()
        targets = ["ws", "ai", "not-bound"]
        _on_hello(kit, InjectedEvent(event=_notice(visibility="all"), target_channel_ids=targets))
        try:
            await _hello(kit)
            assert [e.visibility for e in await _notices(kit)] == ["ai,ws"]
        finally:
            await kit.close()

    async def test_stored_only_is_seen_by_no_channel(self) -> None:
        kit, ws, _ = await _room()
        _on_hello(kit, InjectedEvent(event=_notice(), target_channel_ids=None))
        try:
            await _hello(kit)
            assert [e.visibility for e in await _notices(kit)] == ["none"]
            assert "notice" not in [_body(e) for e in ws.delivered]
        finally:
            await kit.close()

    async def test_stored_only_holds_back_no_later_delivery(self) -> None:
        kit, ws, _ = await _room(delivery_gap_timeout=0.5)
        _on_hello(kit, InjectedEvent(event=_notice(), target_channel_ids=None))
        try:
            await _hello(kit)
            await kit.process_inbound(
                InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="later"))
            )
            assert "later" in [_body(e) for e in ws.delivered]
        finally:
            await kit.close()

    async def test_a_hook_of_its_pass_cannot_widen_its_visibility(self) -> None:
        kit, _, _ = await _room()
        _on_hello(kit, InjectedEvent(event=_notice(), target_channel_ids=["ws"]))

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="widen")
        async def widen(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) != "notice":
                return HookResult.allow()
            return HookResult.modify(event.model_copy(update={"visibility": "all"}))

        try:
            await _hello(kit)
            assert [e.visibility for e in await _notices(kit)] == ["ws"]
        finally:
            await kit.close()

    async def test_carries_no_idempotency_key(self) -> None:
        kit, _, _ = await _room()
        _on_hello(
            kit, InjectedEvent(event=_notice(idempotency_key="k-1"), target_channel_ids=["ws"])
        )
        try:
            await _hello(kit)
            assert [e.idempotency_key for e in await _notices(kit)] == [None]
        finally:
            await kit.close()

    async def test_takes_no_source_check(self) -> None:
        kit, _, _ = await _room()
        kit.register_channel(SimpleChannel("kiosk"))
        await kit.attach_channel("r1", "kiosk", access=Access.READ_ONLY)
        _on_hello(kit, InjectedEvent(event=_notice(source="kiosk"), target_channel_ids=["ws"]))
        try:
            await _hello(kit)
            assert [e.status for e in await _notices(kit)] == [EventStatus.DELIVERED]
        finally:
            await kit.close()


class TestItsDelivery:
    async def test_asks_no_agent_among_its_targets(self) -> None:
        kit, _, ai = await _room()
        _on_hello(kit, InjectedEvent(event=_notice(), target_channel_ids=["ai"]))
        try:
            await _hello(kit)
            assert "notice" not in [_body(e) for e in ai.delivered]
        finally:
            await kit.close()

    async def test_never_reaches_its_own_source(self) -> None:
        kit, ws, _ = await _room()
        _on_hello(kit, InjectedEvent(event=_notice(source="ws"), target_channel_ids=["ws"]))
        try:
            await _hello(kit)
            assert "notice" not in [_body(e) for e in ws.delivered]
        finally:
            await kit.close()

    @pytest.mark.parametrize("targets", [["ws"], None], ids=["delivered", "stored only"])
    async def test_fires_after_broadcast_like_any_other(self, targets: list[str] | None) -> None:
        kit, _, _ = await _room()
        _on_hello(kit, InjectedEvent(event=_notice(), target_channel_ids=targets))
        seen: list[str | None] = []

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="audit")
        async def audit(event: RoomEvent, ctx: RoomContext) -> None:
            seen.append(_body(event))

        try:
            await _hello(kit)
            assert seen.count("notice") == 1
        finally:
            await kit.close()

    async def test_is_transcoded_for_its_target(self) -> None:
        kit, ws, _ = await _room()
        rich = RichContent(body="**ready**", plain_text="ready")
        _on_hello(kit, InjectedEvent(event=_notice(content=rich), target_channel_ids=["ws"]))
        try:
            await _hello(kit)
            assert "ready" in [_body(e) for e in ws.delivered]
        finally:
            await kit.close()


class TestAnIdentityChallenge:
    """The event an identity hook injects takes the same pass (RFC §9.4)."""

    async def _challenged(self, *, blocked: bool) -> tuple[RoomKit, SimpleChannel]:
        kit = RoomKit(identity_resolver=UnknownResolver())
        sms = SimpleChannel("sms")
        kit.register_channel(sms)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms")

        @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
        async def challenge(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            asked = InjectedEvent(event=_notice("Who are you?"), target_channel_ids=["sms"])
            return IdentityHookResult.challenge(inject=asked)

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="no_challenge")
        async def no_challenge(event: RoomEvent, ctx: RoomContext) -> HookResult:
            refused = blocked and _body(event) == "Who are you?"
            return HookResult.block("quiet") if refused else HookResult.allow()

        await kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="hi"))
        )
        return kit, sms

    async def test_names_its_target_as_its_visibility(self) -> None:
        kit, sms = await self._challenged(blocked=False)
        try:
            challenges = await _notices(kit, "Who are you?")
            assert ([e.visibility for e in challenges], len(sms.delivered)) == (["sms"], 1)
        finally:
            await kit.close()

    async def test_runs_the_before_broadcast_hooks(self) -> None:
        kit, sms = await self._challenged(blocked=True)
        try:
            challenges = await _notices(kit, "Who are you?")
            assert ([e.blocked_by for e in challenges], sms.delivered) == (["no_challenge"], [])
        finally:
            await kit.close()


class TestAnotherTrigger:
    """Only BEFORE_BROADCAST and the identity challenge commit injected
    events; any other SYNC trigger drops them, never silently (RFC §9.4)."""

    async def test_drops_them_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        kit, ws, _ = await _room()

        @kit.hook(HookTrigger.BEFORE_DELIVER, name="inject_on_deliver")
        async def inject(event: RoomEvent, ctx: RoomContext) -> HookResult:
            notice = InjectedEvent(event=_notice(), target_channel_ids=["ws"])
            return HookResult.allow(injected=[notice])

        try:
            with caplog.at_level(logging.WARNING, logger="roomkit.hooks"):
                await kit.deliver("r1", "your table is ready")
            assert await _notices(kit) == []
            assert "notice" not in [_body(e) for e in ws.delivered]
            assert any(
                "inject_on_deliver" in r.getMessage() and "before_deliver" in r.getMessage()
                for r in caplog.records
                if r.levelno == logging.WARNING
            )
        finally:
            await kit.close()


class _FilesATask(SimpleChannel):
    """A transport that files a task for every notice it is told about."""

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if _body(event) != "notice":
            return ChannelOutput.empty()
        return ChannelOutput(tasks=[Task(id="read-notice", room_id="r1", title="Read it")])


async def test_its_targets_side_effects_are_collected() -> None:
    kit, _, _ = await _room()
    kit.register_channel(_FilesATask("desk"))
    await kit.attach_channel("r1", "desk")
    _on_hello(kit, InjectedEvent(event=_notice(), target_channel_ids=["desk"]))
    try:
        await _hello(kit)
        assert [task.id for task in await kit.store.list_tasks("r1")] == ["read-notice"]
    finally:
        await kit.close()
