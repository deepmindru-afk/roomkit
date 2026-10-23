"""Hook results and names (RFC §9.1, §9.3, §9.4)."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.core.hooks import RESERVED_HOOK_NAMES, HookEngine, HookRegistration
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, DeleteType, EventType, HookExecution, HookTrigger
from roomkit.models.event import (
    CompositeContent,
    DeleteContent,
    EditContent,
    EventSource,
    RoomEvent,
    TextContent,
)
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter
from roomkit.models.task import Task
from tests.test_framework import AILikeChannel, SimpleChannel

_FINGERPRINT = {"sha256": "0" * 64, "length": 12}
"""What the pipeline writes on a response to an instruction (RFC §10.1.1 step 6)."""


def _context() -> RoomContext:
    return RoomContext(room=Room(id="r1"), bindings=[])


def _hook(fn: Any, name: str = "", *, fail_closed: bool = False) -> HookRegistration:
    return HookRegistration(
        trigger=HookTrigger.BEFORE_BROADCAST,
        execution=HookExecution.SYNC,
        fn=fn,
        name=name,
        fail_closed=fail_closed,
    )


def _event(body: str = "hello", **fields: Any) -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
        content=TextContent(body=body),
        **fields,
    )


async def _run(engine: HookEngine, event: RoomEvent) -> Any:
    return await engine.run_sync_hooks(
        "r1", HookTrigger.BEFORE_BROADCAST, event, _context(), skip_event_filter=True
    )


def _no_payload_modify() -> HookResult:
    # The validator refuses one; a hook can still build it unchecked.
    return HookResult.model_construct(
        action="modify",
        event=None,
        reason=None,
        injected_events=[InjectedEvent(event=_event("notice"))],
        tasks=[Task(id="t1", room_id="r1", title="follow up")],
        observations=[],
        metadata={},
    )


class TestAnUnusableResult:
    async def test_a_modify_without_a_payload_blocks_a_fail_closed_hook(self) -> None:
        engine = HookEngine()

        async def empty(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return _no_payload_modify()

        engine.register(_hook(empty, "empty", fail_closed=True))
        result = await _run(engine, _event())
        assert (result.allowed, result.reason) == (False, "hook_invalid_result:empty")

    async def test_a_wrong_payload_never_reaches_the_next_hook(self) -> None:
        engine = HookEngine()
        seen: list[object] = []

        async def wrong(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult(action="modify", event="not an event")

        async def after(event: object, ctx: RoomContext) -> HookResult:
            seen.append(event)
            return HookResult.allow()

        engine.register(_hook(wrong, "wrong"))
        engine.register(_hook(after, "after"))
        result = await _run(engine, _event())
        assert [type(e).__name__ for e in seen] == ["RoomEvent"]
        assert result.hook_errors and result.allowed

    async def test_keeps_its_tasks_and_drops_its_injected_events(self) -> None:
        engine = HookEngine()

        async def empty(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return _no_payload_modify()

        engine.register(_hook(empty, "empty"))
        result = await _run(engine, _event())
        assert ([t.id for t in result.tasks], result.injected_events) == (["t1"], [])


class TestABeforeBroadcastReplacement:
    async def _replaced(self, original: RoomEvent, **changes: Any) -> RoomEvent:
        engine = HookEngine()

        async def rewrite(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.modify(event.model_copy(update=changes))

        engine.register(_hook(rewrite, "rewrite"))
        result = await _run(engine, original)
        return result.event

    async def test_keeps_which_event_it_is(self) -> None:
        original = _event(chain_depth=2, responds_to="trigger", idempotency_key="key")
        replaced = await self._replaced(
            original,
            id="forged",
            room_id="other-room",
            chain_depth=0,
            responds_to="other",
            idempotency_key="other-key",
            source=EventSource(channel_id="admin", channel_type=ChannelType.SMS),
            content=TextContent(body="redacted"),
        )
        assert (replaced.id, replaced.room_id, replaced.source.channel_id) == (
            original.id,
            "r1",
            "sms",
        )
        assert (replaced.chain_depth, replaced.responds_to, replaced.idempotency_key) == (
            2,
            "trigger",
            "key",
        )
        assert replaced.content == TextContent(body="redacted")

    async def test_keeps_the_event_an_edit_targets(self) -> None:
        original = _event(
            type=EventType.EDIT,
        ).model_copy(
            update={
                "content": EditContent(target_event_id="mine", new_content=TextContent(body="x"))
            }
        )
        replaced = await self._replaced(
            original,
            content=EditContent(target_event_id="theirs", new_content=TextContent(body="y")),
        )
        assert isinstance(replaced.content, EditContent)
        assert replaced.content.target_event_id == "mine"
        assert replaced.content.new_content == TextContent(body="y")

    @pytest.mark.parametrize(
        ("standalone", "written"),
        [({"standalone": True}, {}), ({}, {"standalone": True})],
        ids=["removed", "added"],
    )
    async def test_keeps_an_instruction_s_scope_address_and_standalone_word(
        self, standalone: dict[str, Any], written: dict[str, Any]
    ) -> None:
        original = _event(
            type=EventType.INSTRUCTION,
            visibility="intelligence",
            addressed_to=["ai"],
            metadata={"note": "kept", **standalone},
        )
        replaced = await self._replaced(
            original, visibility="all", addressed_to=[], metadata={"note": "changed", **written}
        )
        assert (replaced.visibility, replaced.addressed_to) == ("intelligence", ["ai"])
        assert replaced.metadata == {"note": "changed", **standalone}

    @pytest.mark.parametrize(
        ("fingerprint", "written"),
        [
            (_FINGERPRINT, {}),
            (_FINGERPRINT, {"instruction": "the instruction's text"}),
            (None, {"instruction": _FINGERPRINT}),
        ],
        ids=["removed", "replaced-by-its-text", "added"],
    )
    async def test_keeps_the_instruction_fingerprint_or_its_absence(
        self, fingerprint: dict[str, Any] | None, written: dict[str, Any]
    ) -> None:
        """RFC §9.4: the fingerprint the pipeline wrote on a response to an
        instruction, or its absence, whatever the hook wrote under its key."""
        kept = {} if fingerprint is None else {"instruction": fingerprint}
        original = _event(responds_to="trigger", metadata=kept)
        replaced = await self._replaced(original, metadata={"note": "added", **written})
        assert replaced.metadata == {"note": "added", **kept}

    async def test_may_change_the_address_of_a_message(self) -> None:
        replaced = await self._replaced(_event(), addressed_to=["billing"])
        assert replaced.addressed_to == ["billing"]


class TestHookNames:
    async def _noop(self, event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult.allow()

    def test_an_explicit_name_is_unique_within_its_scope(self) -> None:
        engine = HookEngine()
        engine.register(_hook(self._noop, "audit"))
        with pytest.raises(ValueError, match="'audit'"):
            engine.register(_hook(self._noop, "audit"))

    def test_a_room_hook_may_reuse_a_kit_hook_s_name(self) -> None:
        engine = HookEngine()
        engine.register(_hook(self._noop, "audit"))
        engine.add_room_hook("r1", _hook(self._noop, "audit"))

    def test_a_default_name_is_made_unique(self) -> None:
        engine = HookEngine()

        async def audit(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.allow()

        first = engine.register(_hook(audit))
        second = engine.register(_hook(audit))
        assert (first.name, second.name) == ("audit", "audit#2")

    @pytest.mark.parametrize("name", sorted(RESERVED_HOOK_NAMES))
    def test_a_name_the_framework_writes_in_blocked_by_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match="blocked_by"):
            HookEngine().register(_hook(self._noop, name))


_EDIT = EditContent(target_event_id="theirs", new_content=TextContent(body="y"))
_DELETE = DeleteContent(target_event_id="theirs")


class TestTheTargetOfAnEditOrADelete:
    async def _replaced(self, original: RoomEvent, content: object) -> RoomEvent:
        engine = HookEngine()

        async def rewrite(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.modify(event.model_copy(update={"content": content}))

        engine.register(_hook(rewrite, "rewrite"))
        return (await _run(engine, original)).event

    async def test_an_edit_rewritten_as_text_stays_the_edit_saying_the_text(self) -> None:
        """A redaction hook that returns text keeps its redaction."""
        original = _event(type=EventType.EDIT).model_copy(
            update={
                "content": EditContent(
                    target_event_id="mine", new_content=TextContent(body="my SSN is 123")
                )
            }
        )
        replaced = await self._replaced(original, TextContent(body="[REDACTED]"))
        assert replaced.content == EditContent(
            target_event_id="mine", new_content=TextContent(body="[REDACTED]")
        )

    async def test_a_delete_keeps_its_target(self) -> None:
        original = _event(type=EventType.DELETE).model_copy(
            update={"content": DeleteContent(target_event_id="mine")}
        )
        replaced = await self._replaced(original, DeleteContent(target_event_id="theirs"))
        assert isinstance(replaced.content, DeleteContent)
        assert replaced.content.target_event_id == "mine"

    async def test_a_change_keeps_the_authority_it_was_checked_with(self) -> None:
        """RFC §9.4, §10.3: a modify keeps the ``edit_source`` and the
        ``delete_type`` step 8a checked; an edit's new content and a
        delete's reason stand."""
        edit = _event(type=EventType.EDIT).model_copy(
            update={
                "content": EditContent(
                    target_event_id="mine",
                    new_content=TextContent(body="x"),
                    edit_source="sender",
                )
            }
        )
        delete = _event(type=EventType.DELETE).model_copy(
            update={"content": DeleteContent(target_event_id="mine")}
        )

        edited = await self._replaced(
            edit,
            EditContent(
                target_event_id="mine", new_content=TextContent(body="y"), edit_source="system"
            ),
        )
        deleted = await self._replaced(
            delete,
            DeleteContent(target_event_id="mine", delete_type=DeleteType.SYSTEM, reason="spam"),
        )

        assert edited.content == EditContent(
            target_event_id="mine", new_content=TextContent(body="y"), edit_source="sender"
        )
        assert deleted.content == DeleteContent(
            target_event_id="mine", delete_type=DeleteType.SENDER, reason="spam"
        )

    @pytest.mark.parametrize(
        ("original_type", "replacement"),
        [
            (EventType.MESSAGE, _EDIT),
            (EventType.MESSAGE, _DELETE),
            (EventType.MESSAGE, CompositeContent(parts=[TextContent(body="y"), _DELETE])),
            (EventType.EDIT, _DELETE),
            (EventType.EDIT, _EDIT.model_copy(update={"new_content": _DELETE})),
            (EventType.EDIT, CompositeContent(parts=[TextContent(body="y"), _DELETE])),
        ],
        ids=[
            "a message made an edit",
            "a message made a delete",
            "a message given a composite holding a delete",
            "an edit made a delete",
            "an edit whose new content is a delete",
            "an edit given a composite holding a delete",
        ],
    )
    async def test_a_change_never_hides_in_a_replacement(
        self, original_type: EventType, replacement: Any
    ) -> None:
        """A replacement that would make a message a change, or hide one in
        an edit's new content, leaves the original's content (RFC §9.4,
        §10.3): the change it holds was never checked."""
        content: Any = TextContent(body="hello")
        if original_type == EventType.EDIT:
            content = _EDIT.model_copy(update={"target_event_id": "mine"})
        original = _event(type=original_type).model_copy(update={"content": content})

        replaced = await self._replaced(original, replacement)

        assert replaced.content == original.content


async def _transport_room(*agents: AILikeChannel) -> tuple[RoomKit, SimpleChannel, SimpleChannel]:
    kit = RoomKit()
    sms, ws = SimpleChannel("sms1"), SimpleChannel("ws1")
    for channel in (sms, ws, *agents):
        kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    for channel in (sms, ws, *agents):
        await kit.attach_channel("r1", channel.channel_id)
    return kit, sms, ws


def _message(body: str, **fields: Any) -> InboundMessage:
    return InboundMessage(
        channel_id="sms1", sender_id="u1", content=TextContent(body=body), **fields
    )


class TestAReplacementInTheRoom:
    async def test_a_hook_cannot_write_under_another_source(self) -> None:
        kit, _sms, _ws = await _transport_room()

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def impersonate(event: RoomEvent, ctx: RoomContext) -> HookResult:
            forged = EventSource(channel_id="ws1", channel_type=ChannelType.SMS)
            return HookResult.modify(event.model_copy(update={"source": forged}))

        try:
            await kit.mute("r1", "sms1")
            result = await kit.process_inbound(_message("hello"))
            assert (result.blocked, result.reason) == (True, "source_muted")
            stored = await kit.store.list_events(
                "r1", event_filter=EventFilter(include_blocked=True)
            )
            assert [e.source.channel_id for e in stored if e.status == "blocked"] == ["sms1"]
        finally:
            await kit.close()

    async def test_an_off_lock_hook_keeps_an_instruction_to_its_agent(self) -> None:
        ai = AILikeChannel("ai")
        kit, _sms, ws = await _transport_room(ai)

        async def scan(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.modify(event.model_copy(update={"metadata": {"scanned": True}}))

        kit.hook(HookTrigger.BEFORE_BROADCAST, needs_lock=False)(scan)
        try:
            await kit.send_event(
                "r1",
                "sms1",
                TextContent(body="Introduce yourself"),
                event_type=EventType.INSTRUCTION,
                addressed_to=["ai"],
            )
            assert "Introduce yourself" not in [_text(e) for e in ws.delivered]
        finally:
            await kit.close()

    async def test_an_off_lock_hook_keeps_a_reply_on_its_thread_root(self) -> None:
        kit, _sms, _ws = await _transport_room()

        async def scan(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.allow()

        kit.hook(HookTrigger.BEFORE_BROADCAST, needs_lock=False)(scan)
        try:
            root = await kit.process_inbound(_message("root"))
            assert root.event is not None
            reply = await kit.process_inbound(_message("reply", parent_event_id=root.event.id))
            assert reply.event is not None
            nested = await kit.process_inbound(_message("nested", parent_event_id=reply.event.id))
            assert nested.event is not None
            assert nested.event.parent_event_id == root.event.id
        finally:
            await kit.close()


def _text(event: RoomEvent) -> str | None:
    return event.content.body if isinstance(event.content, TextContent) else None


class TestTheKitScope:
    async def _noop(self, event: Any, ctx: Any) -> None:
        return None

    def test_one_name_across_the_kit_and_its_executor_is_refused(self) -> None:
        kit = RoomKit()
        kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="audit")(
            self._noop
        )
        with pytest.raises(ValueError, match="'audit'"):
            kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")(
                self._noop
            )

    def test_an_unnamed_turn_hook_is_removed_by_its_function_s_name(self) -> None:
        kit = RoomKit()

        async def audit(event: Any, ctx: Any) -> None:
            return None

        kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)(audit)
        assert kit.hook_engine.remove_global_hook("audit")
        assert kit.turn_executor.hooks._global_hooks == []


class TestReplacingARoomHook:
    async def _first(self, event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult.allow()

    async def _second(self, event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult.allow()

    def test_replaces_the_room_s_hook_of_the_same_name(self) -> None:
        engine = HookEngine()
        engine.add_room_hook("r1", _hook(self._first, "router"))
        engine.replace_room_hook("r1", _hook(self._second, "router"))
        assert [h.fn for h in engine._room_hooks["r1"]] == [self._second]

    def test_a_duplicate_within_one_room_is_refused(self) -> None:
        engine = HookEngine()
        engine.add_room_hook("r1", _hook(self._first, "router"))
        with pytest.raises(ValueError, match="'router'"):
            engine.add_room_hook("r1", _hook(self._second, "router"))
