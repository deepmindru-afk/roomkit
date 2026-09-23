"""Per-hook fail-closed (RFC §9.3): a content check never lets an unchecked payload out."""

from __future__ import annotations

import asyncio

import pytest

from roomkit import RoomKit
from roomkit.core.hooks import HookEngine, HookRegistration
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventStatus, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.room import Room
from tests.conftest import make_event
from tests.test_framework import SimpleChannel


def _ctx() -> RoomContext:
    return RoomContext(room=Room(id="r1"))


async def _slow(event: RoomEvent, ctx: RoomContext) -> HookResult:
    await asyncio.sleep(1)
    return HookResult.allow()


async def _raises(event: RoomEvent, ctx: RoomContext) -> HookResult:
    raise RuntimeError("scanner down")


async def _not_a_result(event: RoomEvent, ctx: RoomContext) -> HookResult:
    return "allow"  # type: ignore[return-value]


async def _modify_with_string(event: RoomEvent, ctx: RoomContext) -> HookResult:
    return HookResult.modify("redacted")  # type: ignore[arg-type]


def _engine(fn: object, *, fail_closed: bool) -> HookEngine:
    engine = HookEngine()
    engine.register(
        HookRegistration(
            trigger=HookTrigger.BEFORE_BROADCAST,
            execution=HookExecution.SYNC,
            fn=fn,  # type: ignore[arg-type]
            name="pii",
            timeout=0.05,
            fail_closed=fail_closed,
        )
    )
    return engine


class TestEngine:
    @pytest.mark.parametrize(
        ("fn", "reason"),
        [
            (_slow, "hook_timeout:pii"),
            (_raises, "hook_error:pii"),
            (_not_a_result, "hook_invalid_result:pii"),
            (_modify_with_string, "hook_invalid_result:pii"),
        ],
    )
    async def test_fail_closed_hook_blocks_with_named_reason(
        self, fn: object, reason: str
    ) -> None:
        engine = _engine(fn, fail_closed=True)
        result = await engine.run_sync_hooks(
            "r1", HookTrigger.BEFORE_BROADCAST, make_event(), _ctx()
        )
        assert result.allowed is False
        assert result.reason == reason
        assert result.blocked_by == "pii"

    @pytest.mark.parametrize("fn", [_slow, _raises, _not_a_result])
    async def test_default_hook_stays_fail_open(self, fn: object) -> None:
        engine = _engine(fn, fail_closed=False)
        result = await engine.run_sync_hooks(
            "r1", HookTrigger.BEFORE_BROADCAST, make_event(), _ctx()
        )
        assert result.allowed is True
        assert result.hook_errors

    async def test_fail_closed_trigger_keeps_its_reason(self) -> None:
        engine = HookEngine()
        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_TTS,
                execution=HookExecution.SYNC,
                fn=_raises,
                name="tts",
            )
        )
        result = await engine.run_sync_hooks(
            "r1", HookTrigger.BEFORE_TTS, "hello", _ctx(), skip_event_filter=True
        )
        assert result.allowed is False
        assert result.reason == "hook tts failed: scanner down"
        assert result.blocked_by is None


class TestInbound:
    async def test_timed_out_check_blocks_the_message(self) -> None:
        kit = RoomKit()
        sender, peer = SimpleChannel("sms1"), SimpleChannel("ws1")
        kit.register_channel(sender)
        kit.register_channel(peer)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")

        kit.hook(
            HookTrigger.BEFORE_BROADCAST, name="pii_detection", timeout=0.05, fail_closed=True
        )(_slow)

        result = await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )

        assert result.blocked is True
        assert result.reason == "hook_timeout:pii_detection"
        assert result.event is not None
        assert result.event.status == EventStatus.BLOCKED
        assert result.event.blocked_by == "pii_detection"
        assert peer.delivered == []

    async def test_room_hook_accepts_fail_closed(self) -> None:
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        kit.add_room_hook(
            "r1",
            HookTrigger.BEFORE_BROADCAST,
            HookExecution.SYNC,
            _raises,
            name="pii",
            fail_closed=True,
        )

        result = await kit.send_event("r1", "sms1", TextContent(body="hi"))

        assert result.status == EventStatus.BLOCKED
        assert result.blocked_by == "pii"
