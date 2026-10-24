"""Visibility, direction and source refusal (RFC §4.2, §7.3, §7.5, §10.2)."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from roomkit import RoomKit, VoiceChannel
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.voice import TTSPlaybackState
from roomkit.channels.websocket import WebSocketChannel
from roomkit.models._visibility import check_channel_id, parse_visibility
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelDirection,
    ChannelType,
    HookTrigger,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.backends.mock import MockVoiceBackend
from tests.test_framework import AILikeChannel, SimpleChannel


class _Told(SimpleChannel):
    """A transport that records what it is told and what it is delivered."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.told: list[RoomEvent] = []
        self.parse_contexts: list[RoomContext] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        self.parse_contexts.append(context)
        return await super().handle_inbound(message, context)

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.told.append(event)
        return ChannelOutput.empty()


def _message(body: str, channel_id: str = "sms", **fields: object) -> InboundMessage:
    return InboundMessage(
        channel_id=channel_id, sender_id="u", content=TextContent(body=body), **fields
    )


class TestTheGrammar:
    def test_a_list_is_written_back_sorted_and_once(self) -> None:
        assert parse_visibility(" ws , sms,ws ") == "sms,ws"

    def test_a_keyword_is_kept_as_written(self) -> None:
        assert parse_visibility(" internal ") == "internal"

    @pytest.mark.parametrize("value", ["", "  ", "sms,,ws", "sms,", "all,sms"])
    def test_an_invalid_value_is_refused(self, value: str) -> None:
        with pytest.raises(ValueError):
            parse_visibility(value)

    @pytest.mark.parametrize("channel_id", ["all", "internal", "a,b", " ws", ""])
    def test_a_channel_id_a_visibility_cannot_name_is_refused(self, channel_id: str) -> None:
        with pytest.raises(ValueError):
            check_channel_id(channel_id)

    def test_a_keyword_in_another_case_is_a_channel_id(self) -> None:
        check_channel_id("All")

    def test_an_inbound_message_refuses_an_empty_visibility(self) -> None:
        with pytest.raises(ValidationError):
            _message("x", visibility="")

    def test_a_channel_named_after_a_keyword_cannot_register(self) -> None:
        with pytest.raises(ValueError):
            RoomKit().register_channel(SimpleChannel("all"))


async def _room(*channels: SimpleChannel | AILikeChannel, **attach: dict) -> RoomKit:
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    for channel in channels:
        kit.register_channel(channel)
        await kit.attach_channel("r1", channel.channel_id, **attach.get(channel.channel_id, {}))
    return kit


class TestTheCommittedVisibility:
    async def test_a_message_s_own_scope_cannot_widen_its_binding(self) -> None:
        sms, ai = _Told("sms"), AILikeChannel("ai")
        kit = await _room(sms, ai, sms={"visibility": "transport"})
        try:
            result = await kit.process_inbound(_message("hello", visibility="all"))
            assert result.event is not None
            assert (result.event.visibility, ai.delivered) == ("transport", [])
        finally:
            await kit.close()

    @pytest.mark.parametrize(
        ("given", "scope", "stored"),
        [("sms", "sms,ws", "sms"), ("transport", "ai,sms", "sms"), ("all", "ws", "ws")],
        ids=["a-list-within-the-scope", "a-category-within-a-list", "unrestricted"],
    )
    async def test_an_answer_keeps_its_channel_s_scope_within_the_message_s(
        self, given: str, scope: str, stored: str
    ) -> None:
        """RFC §5.12, §7.3: what the channel gave its answer is intersected
        with where the message's answers may go, a listed channel's category
        read from the room's bindings: the message's scope never widens it."""
        sms, ws = _Told("sms"), _Told("ws")
        kit = await _room(
            sms, ws, _Says("ai", given), ai={"category": ChannelCategory.INTELLIGENCE}
        )
        try:
            await kit.process_inbound(_message("hello", response_visibility=scope))
            answers = [e for e in await kit.store.list_events("r1") if _body(e) == "AI reply"]
            assert [e.visibility for e in answers] == [stored]
            reached = {
                channel.channel_id
                for channel in (sms, ws)
                if any(_body(e) == "AI reply" for e in channel.delivered)
            }
            assert reached == set(stored.split(","))
        finally:
            await kit.close()

    @pytest.mark.parametrize("scope", [None, "a,b,sms,ws"], ids=["no-scope", "a-wider-scope"])
    async def test_an_answer_s_own_scope_holds_within_the_message_s(
        self, scope: str | None
    ) -> None:
        """RFC §5.12: what answers an answer stays within both the scope the
        answer names and the message's, ``None`` restricting nothing: no
        scope on the message keeps the answer's, a wider one never widens
        it."""
        kit = RoomKit(max_chain_depth=3)
        await kit.create_room(room_id="r1")
        agents = (_Scopes("a", "A", "b,sms"), AILikeChannel("b", response="B"))
        for channel in (_Told("sms"), _Told("ws"), *agents):
            kit.register_channel(channel)
            await kit.attach_channel("r1", channel.channel_id)
        try:
            await kit.process_inbound(_message("hello", response_visibility=scope))
            events = await kit.store.list_events("r1")
            # b's answer to a's answer, which names "b,sms" for its answers.
            assert [e.visibility for e in events if _body(e) == "B" and e.chain_depth == 2] == [
                "b,sms"
            ]
        finally:
            await kit.close()

    async def test_narrowing_a_binding_later_hides_nothing_already_committed(self) -> None:
        sms, ws, ai = _Told("sms"), _Told("ws"), AILikeChannel("ai")
        kit = await _room(sms, ws, ai)
        try:
            await kit.process_inbound(_message("before"))
            await kit.set_visibility("r1", "sms", "ws")
            await kit.process_inbound(_message("after"))
            events = await kit.store.list_events("r1")
            scopes = {
                e.content.body: e.visibility
                for e in events
                if e.source.channel_id == "sms" and isinstance(e.content, TextContent)
            }
            assert scopes == {"before": "all", "after": "ws"}
        finally:
            await kit.close()


class TestDirection:
    async def test_an_inbound_only_transport_is_told_but_not_delivered_to(self) -> None:
        sms, kiosk = _Told("sms"), _Told("kiosk")
        kit = await _room(sms, kiosk, kiosk={"direction": ChannelDirection.INBOUND})
        try:
            await kit.process_inbound(_message("hello"))
            assert (len(kiosk.told), kiosk.delivered) == (1, [])
        finally:
            await kit.close()

    async def test_an_outbound_only_transport_is_delivered_to(self) -> None:
        sms, push = _Told("sms"), _Told("push")
        kit = await _room(sms, push, push={"direction": ChannelDirection.OUTBOUND})
        try:
            await kit.process_inbound(_message("hello"))
            assert (len(push.told), len(push.delivered)) == (1, 1)
        finally:
            await kit.close()

    async def test_direction_does_not_filter_an_agent(self) -> None:
        sms, ai = _Told("sms"), AILikeChannel("ai")
        kit = await _room(
            sms,
            ai,
            ai={"category": ChannelCategory.INTELLIGENCE, "direction": ChannelDirection.OUTBOUND},
        )
        try:
            await kit.process_inbound(_message("hello"))
            assert len(ai.delivered) == 1
        finally:
            await kit.close()


class TestARefusedSource:
    async def test_read_only_wins_over_muted(self) -> None:
        sms, ws = _Told("sms"), _Told("ws")
        kit = await _room(sms, ws, sms={"access": Access.READ_ONLY})
        try:
            await kit.mute("r1", "sms")
            result = await kit.process_inbound(_message("hello"))
            assert (result.blocked, result.reason) == (True, "source_read_only")
        finally:
            await kit.close()

    async def test_a_detached_channel_s_message_is_stored_refused(self) -> None:
        sms, ws = _Told("sms"), _Told("ws")
        kit = await _room(sms, ws)
        try:
            await kit.process_inbound(_message("before"))
            await kit.detach_channel("r1", "sms")
            result = await kit.process_inbound(_message("after"), room_id="r1")
            assert (result.blocked, result.reason) == (True, "source_unbound")
            assert sms.parse_contexts[-1].recent_events == []
            stored = await kit.store.list_events(
                "r1", event_filter=EventFilter(include_blocked=True)
            )
            assert [e.blocked_by for e in stored if _body(e) == "after"] == ["source_unbound"]
            assert [_body(e) for e in ws.delivered] == ["before"]
        finally:
            await kit.close()


def _body(event: RoomEvent) -> str | None:
    return event.content.body if isinstance(event.content, TextContent) else None


class _Chatty(AILikeChannel):
    """An agent answering every other channel's event with *count* answers."""

    def __init__(self, channel_id: str, count: int) -> None:
        super().__init__(channel_id)
        self._count = count

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        source = EventSource(channel_id=self.channel_id, channel_type=ChannelType.AI)
        answers = [
            RoomEvent(room_id=event.room_id, source=source, content=TextContent(body=f"A{i}"))
            for i in range(self._count)
        ]
        return ChannelOutput(responded=True, response_events=answers)


async def _blocked_scopes(kit: RoomKit, channel_id: str) -> list[str]:
    events = await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))
    return [
        e.visibility
        for e in events
        if e.source.channel_id == channel_id and e.blocked_by is not None
    ]


class TestARefusedRecordSeesNoMoreThanItsEvent:
    """RFC §7.3, §8.3: a record stored BLOCKED takes the visibility its event
    would have been delivered with, its own intersected with its source
    binding's, so a reader shown it later sees no more of it than of the
    event."""

    @pytest.mark.parametrize("refusal", ["a hook", "its source"])
    @pytest.mark.parametrize("entry", ["process_inbound", "send_event"])
    async def test_a_refused_message(self, refusal: str, entry: str) -> None:
        sms, ws = _Told("sms"), _Told("ws")
        binding = {"visibility": "ws"}
        if refusal == "its source":
            binding["access"] = Access.READ_ONLY
        kit = await _room(sms, ws, sms=binding)
        if refusal == "a hook":
            kit.hook(HookTrigger.BEFORE_BROADCAST, name="gate")(
                lambda event, ctx: _blocking(event)
            )
        try:
            if entry == "process_inbound":
                await kit.process_inbound(_message("hello"))
            else:
                await kit.send_event("r1", "sms", TextContent(body="hello"))
            assert await _blocked_scopes(kit, "sms") == ["ws"]
        finally:
            await kit.close()

    async def test_the_record_of_an_agent_not_asked_at_the_depth_limit(self) -> None:
        kit = RoomKit(max_chain_depth=1)
        await kit.create_room(room_id="r1")
        for channel in (_Told("sms"), _Told("ws"), AILikeChannel("ai")):
            kit.register_channel(channel)
        await kit.attach_channel("r1", "sms")
        await kit.attach_channel("r1", "ws")
        await kit.attach_channel(
            "r1", "ai", category=ChannelCategory.INTELLIGENCE, visibility="ws"
        )
        try:
            await kit.process_inbound(_message("hello"))
            assert await _blocked_scopes(kit, "ai") == ["ws"]
        finally:
            await kit.close()

    async def test_an_answer_stored_past_the_reentry_budget(self) -> None:
        """A budget of 20 (``max_chain_depth=2``): the 21st answer is capped."""
        kit = RoomKit(max_chain_depth=2)
        await kit.create_room(room_id="r1")
        for channel in (_Told("sms"), _Told("ws"), _Chatty("ai", count=21)):
            kit.register_channel(channel)
        await kit.attach_channel("r1", "sms")
        await kit.attach_channel("r1", "ws")
        await kit.attach_channel(
            "r1", "ai", category=ChannelCategory.INTELLIGENCE, visibility="ws"
        )
        try:
            await kit.process_inbound(_message("hello"))
            assert await _blocked_scopes(kit, "ai") == ["ws"]
        finally:
            await kit.close()

    async def test_an_event_a_hook_injects_keeps_its_own(self) -> None:
        """No channel's: blocked by a hook of its pass, it keeps its targets."""
        sms, ws = _Told("sms"), _Told("ws")
        kit = await _room(sms, ws, sms={"visibility": "sms"})

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="notify")
        async def notify(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) != "hello":
                return HookResult.block("held") if _body(event) == "notice" else HookResult.allow()
            notice = RoomEvent(
                room_id="r1", source=event.source, content=TextContent(body="notice")
            )
            return HookResult.allow([InjectedEvent(event=notice, target_channel_ids=["ws"])])

        try:
            await kit.process_inbound(_message("hello"))
            assert await _blocked_scopes(kit, "sms") == ["ws"]
        finally:
            await kit.close()


async def _blocking(event: RoomEvent) -> HookResult:
    return HookResult.block("held")


class TestTheEntriesRefuseAnInvalidVisibility:
    async def test_set_visibility_refuses_an_empty_item(self) -> None:
        kit = await _room(_Told("sms"))
        try:
            with pytest.raises(ValueError):
                await kit.set_visibility("r1", "sms", "sms,,ws")
        finally:
            await kit.close()

    async def test_attach_channel_refuses_an_empty_visibility(self) -> None:
        kit = await _room(_Told("sms"))
        kit.register_channel(_Told("ws"))
        try:
            with pytest.raises(ValueError):
                await kit.attach_channel("r1", "ws", visibility="")
        finally:
            await kit.close()

    async def test_send_event_refuses_a_keyword_in_a_list(self) -> None:
        kit = await _room(_Told("sms"))
        try:
            with pytest.raises(ValueError):
                await kit.send_event("r1", "sms", TextContent(body="x"), visibility="all,sms")
        finally:
            await kit.close()


class TestEveryCommitPathStampsTheVisibility:
    async def test_a_greeting_takes_its_agent_binding_s_scope(self) -> None:
        kit = RoomKit()
        kit.register_channel(SimpleChannel("advisor"))
        kit.register_channel(SimpleChannel("customer"))
        kit.register_channel(
            Agent("ai", provider=MockAIProvider(responses=["x"]), greeting="Hi!", auto_greet=False)
        )
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "advisor")
        await kit.attach_channel("r1", "customer")
        await kit.attach_channel(
            "r1", "ai", category=ChannelCategory.INTELLIGENCE, visibility="advisor"
        )
        try:
            await kit.send_greeting("r1")
            events = await kit.store.list_events("r1")
            greetings = [e for e in events if (e.metadata or {}).get("auto_greeting")]
            assert [e.visibility for e in greetings] == ["advisor"]
        finally:
            await kit.close()

    async def test_an_injected_event_takes_its_source_binding_s_scope(self) -> None:
        sms, ws, other = _Told("sms"), _Told("ws"), _Told("other")
        kit = await _room(sms, ws, other, sms={"visibility": "ws"})
        notice = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
            content=TextContent(body="notice"),
        )

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def inject(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) != "hello":
                return HookResult.allow()
            return HookResult.allow(
                injected=[InjectedEvent(event=notice, target_channel_ids=["ws"])]
            )

        try:
            await kit.process_inbound(_message("hello"))
            events = await kit.store.list_events("r1")
            assert [e.visibility for e in events if _body(e) == "notice"] == ["ws"]
        finally:
            await kit.close()


async def _streaming_room(
    agent_binding: dict, *transports: tuple[str, dict]
) -> tuple[RoomKit, dict[str, list[str]]]:
    """A streaming agent answering an SMS sender, and WebSocket transports
    that record the chunks they are streamed."""
    kit = RoomKit()
    provider = MockAIProvider(
        streaming=True, ai_responses=[AIResponse(content="answer", finish_reason="stop")]
    )
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(AIChannel("ai", provider=provider))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    streamed: dict[str, list[str]] = {}
    for channel_id, binding in transports:
        ws = WebSocketChannel(channel_id)
        chunks = streamed.setdefault(channel_id, [])

        async def send_fn(conn_id: str, event: RoomEvent) -> None:
            pass

        async def stream_send_fn(conn_id: str, msg: object, chunks: list[str] = chunks) -> None:
            delta = getattr(msg, "delta", None)
            if delta:
                chunks.append(delta)

        ws.register_connection("c1", send_fn, stream_send_fn=stream_send_fn, room_id="r1")
        kit.register_channel(ws)
        await kit.attach_channel("r1", channel_id, **binding)
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE, **agent_binding)
    return kit, streamed


class TestALiveStream:
    async def test_reaches_no_transport_the_agent_s_binding_excludes(self) -> None:
        kit, streamed = await _streaming_room(
            {"visibility": "advisor"}, ("customer", {}), ("advisor", {})
        )
        try:
            await kit.process_inbound(_message("hello"))
            assert streamed["customer"] == []
        finally:
            await kit.close()

    async def test_reaches_no_inbound_only_transport(self) -> None:
        kit, streamed = await _streaming_room(
            {}, ("kiosk", {"direction": ChannelDirection.INBOUND})
        )
        try:
            await kit.process_inbound(_message("hello"))
            assert streamed["kiosk"] == []
        finally:
            await kit.close()

    async def test_reaches_an_outbound_only_transport(self) -> None:
        kit, streamed = await _streaming_room(
            {}, ("push", {"direction": ChannelDirection.OUTBOUND})
        )
        try:
            await kit.process_inbound(_message("hello"))
            assert "".join(streamed["push"]) == "answer"
        finally:
            await kit.close()


class _Scopes(AILikeChannel):
    """An agent whose answers name *scope* for what answers them."""

    def __init__(self, channel_id: str, response: str, scope: str) -> None:
        super().__init__(channel_id, response=response)
        self._scope = scope

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        output = await super().on_event(event, binding, context)
        scoped = [
            e.model_copy(update={"response_visibility": self._scope})
            for e in output.response_events
        ]
        return output.model_copy(update={"response_events": scoped})


class _Says(AILikeChannel):
    """An agent whose answer carries *visibility*."""

    def __init__(self, channel_id: str, visibility: str) -> None:
        super().__init__(channel_id)
        self._visibility = visibility

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        output = await super().on_event(event, binding, context)
        marked = [
            e.model_copy(update={"visibility": self._visibility}) for e in output.response_events
        ]
        return output.model_copy(update={"response_events": marked})


class TestAnEmptyAnswerScope:
    """RFC §5.12: ``None`` alone restricts nothing; any other scope is one,
    an empty one naming nobody, on every path (fail-closed)."""

    @pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streamed"])
    async def test_a_hook_writing_an_empty_scope_sends_the_answers_nowhere(
        self, streamed: bool
    ) -> None:
        if streamed:
            kit, _ = await _streaming_room({})
            body = "answer"
        else:
            kit = await _room(
                _Told("sms"), AILikeChannel("ai"), ai={"category": ChannelCategory.INTELLIGENCE}
            )
            body = "AI reply"

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="empty-scope")
        async def scope(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.source.channel_id != "sms":
                return HookResult.allow()
            return HookResult.modify(event.model_copy(update={"response_visibility": ""}))

        try:
            await kit.process_inbound(_message("hello"))
            answers = [e for e in await kit.store.list_events("r1") if _body(e) == body]
            assert [e.visibility for e in answers] == ["none"]
        finally:
            await kit.close()


def _scope_answers_internal(kit: RoomKit) -> None:
    """A hook that scopes the answers to the message ``internal``, which no
    door takes (RFC §7.3)."""

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="scope-answers")
    async def scope(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.source.channel_id != "sms":
            return HookResult.allow()
        return HookResult.modify(event.model_copy(update={"response_visibility": "internal"}))


class TestOnlyTheFrameworkWritesInternal:
    """RFC §7.3: ``internal`` marks a record the framework wrote; a door
    refuses it before anything is read or written, a hook's modify cannot
    write it, and a channel's answer carrying it is stored ``none``."""

    @pytest.mark.parametrize(
        "door",
        [
            "attach",
            "binding update",
            "inbound",
            "injection",
            "inbound answers",
            "injection answers",
        ],
    )
    async def test_a_door_refuses_it_with_nothing_written(self, door: str) -> None:
        kit = await _room(_Told("sms"))
        kit.register_channel(_Told("ws"))
        enters = {
            "attach": lambda: kit.attach_channel("r1", "ws", visibility="internal"),
            "binding update": lambda: kit.set_visibility("r1", "sms", "internal"),
            "inbound": lambda: kit.process_inbound(_message("x", visibility="internal")),
            "injection": lambda: kit.send_event(
                "r1", "sms", TextContent(body="x"), visibility="internal"
            ),
            # Where the answers may go is never the framework's mark either.
            "inbound answers": lambda: kit.process_inbound(
                _message("x", response_visibility="internal")
            ),
            "injection answers": lambda: kit.send_event(
                "r1", "sms", TextContent(body="x"), response_visibility="internal"
            ),
        }
        before = await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))
        try:
            with pytest.raises((ValueError, ValidationError), match="internal"):
                await enters[door]()
            after = await kit.store.list_events(
                "r1", event_filter=EventFilter(include_blocked=True)
            )
            bindings = {b.channel_id: b.visibility for b in await kit.store.list_bindings("r1")}
            assert (len(after), bindings) == (len(before), {"sms": "all"})
        finally:
            await kit.close()

    async def test_a_hook_writing_it_has_the_visibility_put_back(self) -> None:
        kit = await _room(_Told("sms"))

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="hide")
        async def hide(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.modify(event.model_copy(update={"visibility": "internal"}))

        try:
            result = await kit.process_inbound(_message("hello", visibility="sms"))
            assert result.event is not None
            assert result.event.visibility == "sms"
        finally:
            await kit.close()

    async def test_a_hook_cannot_widen_the_framework_s_own_record(self) -> None:
        backend = MockVoiceBackend()
        kit = RoomKit(voice=backend)
        voice = VoiceChannel("voice", backend=backend)
        kit.register_channel(voice)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "voice")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="widen")
        async def widen(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.modify(event.model_copy(update={"visibility": "all"}))

        try:
            session = await kit.connect_voice("r1", "user-1", "voice")
            # A barge-in's record of what was heard (RFC §6.4).
            cut = TTSPlaybackState(session.id, "Rain tomorrow.", answer_channel_id="ai")
            await voice._store_interrupted_utterance(session, cut, "r1")
            records = [
                e for e in await kit.store.list_events("r1") if e.metadata.get("interrupted")
            ]
            assert [e.visibility for e in records] == ["internal"]
        finally:
            await kit.close()

    async def test_an_answer_carrying_it_is_stored_for_no_channel(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        kit = await _room(
            _Told("sms"), _Says("ai", "internal"), ai={"category": ChannelCategory.INTELLIGENCE}
        )
        try:
            with caplog.at_level(logging.WARNING):
                await kit.process_inbound(_message("hello"))
            answers = [e for e in await kit.store.list_events("r1") if _body(e) == "AI reply"]
            assert [e.visibility for e in answers] == ["none"]
            assert any("internal" in r.getMessage() for r in caplog.records)
        finally:
            await kit.close()

    async def test_the_record_in_place_of_an_answer_carrying_it_is_stored_for_no_channel(
        self,
    ) -> None:
        kit = RoomKit(max_chain_depth=1)
        await kit.create_room(room_id="r1")
        for channel in (_Told("sms"), AILikeChannel("ai")):
            kit.register_channel(channel)
        await kit.attach_channel("r1", "sms")
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        _scope_answers_internal(kit)
        try:
            # The depth limit keeps the agent from being asked: its record
            # takes the answer's scope.
            await kit.process_inbound(_message("hello"))
            records = await kit.store.list_events(
                "r1", event_filter=EventFilter(include_blocked=True)
            )
            assert [e.visibility for e in records if e.source.channel_id == "ai"] == ["none"]
        finally:
            await kit.close()

    async def test_a_stream_row_carrying_it_is_stored_for_no_channel(self) -> None:
        kit, _ = await _streaming_room({})
        _scope_answers_internal(kit)
        try:
            await kit.process_inbound(_message("hello"))
            answers = [e for e in await kit.store.list_events("r1") if _body(e) == "answer"]
            assert [e.visibility for e in answers] == ["none"]
        finally:
            await kit.close()
