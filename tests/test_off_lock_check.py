"""Off-lock BEFORE_BROADCAST checks (``needs_lock=False``, RFC §9.5.1)."""

from __future__ import annotations

import asyncio

import pytest

from roomkit import RoomKit
from roomkit.core.hooks import HookEngine, HookRegistration
from roomkit.core.locks import _has_room_lock
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelType,
    EventStatus,
    EventType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.store_filter import EventFilter
from tests.test_framework import AILikeChannel, SimpleChannel


def _body(event: RoomEvent) -> str:
    return getattr(event.content, "body", "")


def _msg(body: str) -> InboundMessage:
    return InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body=body))


async def _room(kit: RoomKit) -> SimpleChannel:
    peer = SimpleChannel("ws1")
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(peer)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ws1")
    return peer


def _scan(delays: dict[str, float], *, block: set[str] | None = None):  # noqa: ANN202
    """An off-lock check whose duration (and verdict) depends on the body."""

    async def scan(event: RoomEvent, ctx: RoomContext) -> HookResult:
        body = _body(event)
        await asyncio.sleep(delays.get(body, 0))
        if block and body in block:
            return HookResult.block("pii")
        return HookResult.allow()

    return scan


class TestOrderAndOverlap:
    async def test_checks_overlap_and_commits_keep_arrival_order(self) -> None:
        kit = RoomKit()
        peer = await _room(kit)
        # A's check is slower than B's: B finishes first, still commits second.
        kit.hook(HookTrigger.BEFORE_BROADCAST, name="pii", needs_lock=False)(
            _scan({"A": 0.4, "B": 0.3})
        )

        loop = asyncio.get_running_loop()
        start = loop.time()
        first = asyncio.create_task(kit.process_inbound(_msg("A"), room_id="r1"))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(kit.process_inbound(_msg("B"), room_id="r1"))
        a, b = await asyncio.gather(first, second)
        elapsed = loop.time() - start

        assert a.event is not None and b.event is not None
        assert a.event.index < b.event.index
        assert [_body(e) for e in peer.delivered] == ["A", "B"]
        # Serialized under the lock this is 0.7s; overlapped it is A's 0.4s.
        assert elapsed < 0.6
        assert kit._admission.pending("r1") == 0

    async def test_locked_hook_still_serializes(self) -> None:
        kit = RoomKit()
        await _room(kit)
        kit.hook(HookTrigger.BEFORE_BROADCAST, name="pii")(_scan({"A": 0.2, "B": 0.2}))

        loop = asyncio.get_running_loop()
        start = loop.time()
        await asyncio.gather(
            kit.process_inbound(_msg("A"), room_id="r1"),
            kit.process_inbound(_msg("B"), room_id="r1"),
        )
        assert loop.time() - start >= 0.39

    async def test_blocked_first_message_releases_the_next(self) -> None:
        kit = RoomKit()
        peer = await _room(kit)
        kit.hook(HookTrigger.BEFORE_BROADCAST, name="pii", needs_lock=False)(
            _scan({"A": 0.1}, block={"A"})
        )

        a, b = await asyncio.gather(
            kit.process_inbound(_msg("A"), room_id="r1"),
            kit.process_inbound(_msg("B"), room_id="r1"),
        )

        assert a.blocked and a.event is not None
        assert a.event.status == EventStatus.BLOCKED
        assert a.event.blocked_by == "pii"
        assert not b.blocked
        assert [_body(e) for e in peer.delivered] == ["B"]

    async def test_direct_injection_takes_the_same_path(self) -> None:
        kit = RoomKit()
        await _room(kit)
        seen: list[bool] = []

        async def check(event: RoomEvent, ctx: RoomContext) -> HookResult:
            seen.append(_has_room_lock("r1", kit._lock_manager))
            return HookResult.block("pii")

        kit.hook(HookTrigger.BEFORE_BROADCAST, name="pii", needs_lock=False)(check)

        event = await kit.send_event("r1", "sms1", TextContent(body="secret"))

        assert event.status == EventStatus.BLOCKED
        assert seen == [False]


class TestDecisionMerge:
    async def test_locked_hooks_see_the_off_lock_modification(self) -> None:
        kit = RoomKit()
        peer = await _room(kit)
        seen: list[str] = []

        async def redact(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.modify(event.model_copy(update={"content": TextContent(body="***")}))

        async def audit(event: RoomEvent, ctx: RoomContext) -> HookResult:
            seen.append(_body(event))
            return HookResult.allow()

        kit.hook(HookTrigger.BEFORE_BROADCAST, name="pii", priority=-1, needs_lock=False)(redact)
        kit.hook(HookTrigger.BEFORE_BROADCAST, name="audit")(audit)

        await kit.process_inbound(_msg("my card is 4111"), room_id="r1")

        assert seen == ["***"]
        assert [_body(e) for e in peer.delivered] == ["***"]

    async def test_async_observers_fire_once(self) -> None:
        kit = RoomKit()
        await _room(kit)
        fired: list[str] = []

        async def observe(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(_body(event))

        kit.hook(HookTrigger.BEFORE_BROADCAST, name="pii", needs_lock=False)(_scan({}))
        kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.ASYNC, name="obs")(observe)

        await kit.process_inbound(_msg("hi"), room_id="r1")
        await asyncio.sleep(0.05)

        assert fired == ["hi"]

    async def test_other_commit_paths_still_run_the_check(self) -> None:
        """A reentry commit (the AI's answer) has no off-lock phase: the
        off-lock hook runs there under the lock, it is never skipped."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(AILikeChannel("ai1", response="leak"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ai1")
        kit.hook(HookTrigger.BEFORE_BROADCAST, name="pii", needs_lock=False)(
            _scan({}, block={"leak"})
        )

        await kit.process_inbound(_msg("hi"), room_id="r1")

        events = await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))
        leaked = [e for e in events if _body(e) == "leak"]
        assert leaked and all(e.status == EventStatus.BLOCKED for e in leaked)


class TestTicketRelease:
    async def _next_goes_through(self, kit: RoomKit) -> None:
        # The hook stays registered: the next message takes a ticket too, and
        # would hang behind a leaked one.
        assert kit._admission.pending("r1") == 0
        result = await asyncio.wait_for(kit.process_inbound(_msg("next"), room_id="r1"), 1)
        assert not result.blocked

    async def test_released_when_the_check_raises(self) -> None:
        kit = RoomKit()
        await _room(kit)

        async def boom(event: RoomEvent, ctx: RoomContext) -> HookResult:
            raise RuntimeError("scanner down")

        kit.hook(HookTrigger.BEFORE_BROADCAST, name="pii", needs_lock=False)(boom)
        await kit.process_inbound(_msg("A"), room_id="r1")
        await self._next_goes_through(kit)

    async def test_released_when_the_check_times_out_closed(self) -> None:
        kit = RoomKit()
        await _room(kit)
        kit.hook(
            HookTrigger.BEFORE_BROADCAST,
            name="pii",
            timeout=0.05,
            fail_closed=True,
            needs_lock=False,
        )(_scan({"A": 1}))

        result = await kit.process_inbound(_msg("A"), room_id="r1")

        assert result.reason == "hook_timeout:pii"
        await self._next_goes_through(kit)

    async def test_released_when_process_timeout_expires(self) -> None:
        kit = RoomKit(process_timeout=0.1)
        await _room(kit)
        kit.hook(HookTrigger.BEFORE_BROADCAST, name="pii", needs_lock=False)(_scan({"A": 1}))

        result = await kit.process_inbound(_msg("A"), room_id="r1")

        assert result.blocked and result.reason == "process_timeout"
        events = await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))
        assert [_body(e) for e in events if e.type == EventType.MESSAGE] == []
        await self._next_goes_through(kit)

    async def test_released_when_the_caller_cancels(self) -> None:
        kit = RoomKit()
        await _room(kit)
        kit.hook(HookTrigger.BEFORE_BROADCAST, name="pii", needs_lock=False)(_scan({"A": 1}))

        task = asyncio.create_task(kit.process_inbound(_msg("A"), room_id="r1"))
        await asyncio.sleep(0.05)
        assert kit._admission.pending("r1") == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await self._next_goes_through(kit)


class TestReentrance:
    async def test_event_injected_from_the_check_does_not_deadlock(self) -> None:
        kit = RoomKit()
        await _room(kit)
        kit.register_channel(SimpleChannel("sys", ChannelType.SYSTEM))
        await kit.attach_channel("r1", "sys")

        async def scan_and_notify(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.source.channel_id == "sms1":
                await kit.send_event("r1", "sys", TextContent(body="PII check running"))
            return HookResult.allow()

        kit.hook(HookTrigger.BEFORE_BROADCAST, name="pii", needs_lock=False)(scan_and_notify)

        result = await asyncio.wait_for(kit.process_inbound(_msg("A"), room_id="r1"), 2)

        events = await kit.store.list_events("r1")
        notice = next(e for e in events if _body(e) == "PII check running")
        assert result.event is not None
        assert notice.index < result.event.index


def _reg(
    name: str, *, priority: int = 0, needs_lock: bool = True, **kw: object
) -> HookRegistration:
    async def fn(event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult.allow()

    return HookRegistration(
        trigger=kw.pop("trigger", HookTrigger.BEFORE_BROADCAST),  # type: ignore[arg-type]
        execution=kw.pop("execution", HookExecution.SYNC),  # type: ignore[arg-type]
        fn=fn,
        name=name,
        priority=priority,
        needs_lock=needs_lock,
    )


class TestPlacementRules:
    @pytest.mark.parametrize(
        "kw",
        [
            {"trigger": HookTrigger.AFTER_BROADCAST},
            {"execution": HookExecution.ASYNC},
        ],
    )
    def test_off_lock_only_on_sync_before_broadcast(self, kw: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="only supported"):
            HookEngine().register(_reg("pii", needs_lock=False, **kw))

    def test_locked_hook_ordered_before_off_lock_is_refused(self) -> None:
        engine = HookEngine()
        engine.register(_reg("consent", priority=-10))
        with pytest.raises(ValueError, match="'consent'.*'pii'"):
            engine.register(_reg("pii", priority=0, needs_lock=False))

    def test_refused_in_either_registration_order(self) -> None:
        engine = HookEngine()
        engine.register(_reg("pii", priority=0, needs_lock=False))
        with pytest.raises(ValueError, match="'router'.*'pii'"):
            engine.add_room_hook("r1", _reg("router", priority=-100))

    def test_off_lock_at_or_below_the_locked_priority_is_accepted(self) -> None:
        engine = HookEngine()
        engine.add_room_hook("r1", _reg("router", priority=-100))
        engine.register(_reg("consent", priority=-210, needs_lock=False))
        engine.register(_reg("pii", priority=-200, needs_lock=False))
        engine.register(_reg("tie", priority=-100, needs_lock=False))

    def test_hooks_of_other_rooms_do_not_meet(self) -> None:
        engine = HookEngine()
        engine.add_room_hook("r1", _reg("pii", priority=0, needs_lock=False))
        engine.add_room_hook("r2", _reg("router", priority=-100))


class TestReentranceUnderTheLock:
    async def test_event_injected_by_a_locked_hook_does_not_wait_on_its_caller(self) -> None:
        """A locked hook holds the caller's ticket AND the lock: an event it
        injects must not queue behind that ticket (RFC §9.5.1)."""
        kit = RoomKit()
        await _room(kit)
        kit.hook(HookTrigger.BEFORE_BROADCAST, name="pii", priority=-10, needs_lock=False)(
            _scan({})
        )

        async def notify(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) == "A":
                await kit.send_event("r1", "sms1", TextContent(body="notice"))
            return HookResult.allow()

        kit.hook(HookTrigger.BEFORE_BROADCAST, name="notify")(notify)

        result = await asyncio.wait_for(kit.process_inbound(_msg("A"), room_id="r1"), 2)

        assert not result.blocked
        events = await kit.store.list_events("r1")
        assert [_body(e) for e in events if e.type == EventType.MESSAGE] == ["notice", "A"]


def test_fail_closed_is_refused_on_an_async_hook() -> None:
    reg = _reg("audit", execution=HookExecution.ASYNC)
    reg.fail_closed = True
    with pytest.raises(ValueError, match="fail_closed"):
        HookEngine().register(reg)
