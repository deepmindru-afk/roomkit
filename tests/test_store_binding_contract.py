"""Binding serialization contract shared by all conversation stores, and the
timeline rules every store keeps alike."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from roomkit.core.exceptions import EventNotFoundError
from roomkit.core.inbound_router import DefaultInboundRoomRouter
from roomkit.models.channel import ChannelBinding, RateLimit, RetryPolicy
from roomkit.models.enums import ChannelType, EventStatus, EventType, RoomStatus
from roomkit.models.event import DeleteContent, EditContent, RoomEvent, TextContent
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import SQLiteStore
from tests.conftest import make_event


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def contract_store(request: pytest.FixtureRequest) -> AsyncIterator[ConversationStore]:
    store: ConversationStore
    if request.param == "memory":
        store = InMemoryStore()
    elif request.param == "sqlite":
        store = SQLiteStore(":memory:")
    else:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            pytest.skip("POSTGRES_DSN not set")
        from roomkit.store.postgres import PostgresStore

        store = PostgresStore(dsn=dsn)
        await store.init(min_size=1, max_size=2)
    try:
        yield store
    finally:
        await store.close()


async def test_binding_round_trip_and_policy_updates(contract_store: ConversationStore) -> None:
    room = await contract_store.create_room(Room(id=uuid4().hex))
    binding = ChannelBinding(
        room_id=room.id,
        channel_id="sms",
        channel_type="sms",
        access="read_only",
        muted=True,
        output_muted=True,
        visibility="private",
        participant_id="alice",
        group=True,
        last_read_index=4,
        metadata={"nested": {"key": [1]}},
        rate_limit=RateLimit(max_per_minute=30),
        retry_policy=RetryPolicy(max_retries=7, base_delay_seconds=0.3),
    )
    try:
        await contract_store.add_binding(binding)
        assert await contract_store.get_binding(room.id, "sms") == binding
        assert await contract_store.list_bindings(room.id) == [binding]
        changed = binding.model_copy(update={"rate_limit": RateLimit(max_per_second=2)})
        await contract_store.add_binding(changed)
        assert await contract_store.get_binding(room.id, "sms") == changed
        changed = changed.model_copy(update={"retry_policy": RetryPolicy(max_retries=2)})
        await contract_store.update_binding(changed)
        assert await contract_store.get_binding(room.id, "sms") == changed
        cleared = changed.model_copy(update={"rate_limit": None, "retry_policy": None})
        await contract_store.update_binding(cleared)
        assert await contract_store.get_binding(room.id, "sms") == cleared
    finally:
        await contract_store.delete_room(room.id)


async def test_binding_exists_answers_as_get_binding_does(
    contract_store: ConversationStore,
) -> None:
    """Routing's yes/no (RFC §10.1 step 2) is the one ``get_binding`` would give."""
    room = await contract_store.create_room(Room(id=uuid4().hex))
    try:
        assert not await contract_store.binding_exists(room.id, "sms")
        await contract_store.add_binding(
            ChannelBinding(room_id=room.id, channel_id="sms", channel_type=ChannelType.SMS)
        )
        assert await contract_store.binding_exists(room.id, "sms")
        assert not await contract_store.binding_exists(room.id, "email")
        assert not await contract_store.binding_exists(uuid4().hex, "sms")
        await contract_store.remove_binding(room.id, "sms")
        assert not await contract_store.binding_exists(room.id, "sms")
    finally:
        await contract_store.delete_room(room.id)


@pytest.mark.parametrize("newest_first", [False, True])
@pytest.mark.parametrize("cursor", ["after", "before", "offset"])
async def test_pagination_order_contract(
    contract_store: ConversationStore, newest_first: bool, cursor: str
) -> None:
    """A cursor or ``newest_first`` selects the *window*; the page is always
    rendered ascending (RFC §14.1). ``before_index`` and the newest-first
    offset page both come back oldest-first, so ``page[-1]`` is the newest
    event of the window on every backend, never ``page[0]``. The ``offset=99``
    under a cursor asserts the other half of the rule: a cursor ignores
    ``offset``."""
    room = await contract_store.create_room(Room(id=uuid4().hex))
    try:
        for i in range(6):
            await contract_store.add_event(make_event(room_id=room.id, index=i, body=str(i)))
        if cursor == "after":
            page = await contract_store.list_events(
                room.id, after_index=1, newest_first=newest_first, limit=2, offset=99
            )
            expected = [2, 3]
        elif cursor == "before":
            page = await contract_store.list_events(
                room.id, before_index=4, newest_first=newest_first, limit=2, offset=99
            )
            expected = [2, 3]
        else:
            page = await contract_store.list_events(
                room.id, newest_first=newest_first, limit=2, offset=1
            )
            expected = [3, 4] if newest_first else [1, 2]
        assert [event.index for event in page] == expected
    finally:
        await contract_store.delete_room(room.id)


async def test_refused_rows_are_served_on_request_only(contract_store: ConversationStore) -> None:
    """RFC §14.1: a timeline read serves what the room received. A row stored
    BLOCKED is skipped by default, before the page is cut; ``include_blocked``
    lifts the filter, ``get_event`` never applies it, and ``get_conversation``
    reads the whole conversation because hooks read it whole (§7.5 rule 8)."""
    room = await contract_store.create_room(Room(id=uuid4().hex))
    try:
        root = make_event(room_id=room.id, index=0, body="0")
        refused = make_event(
            room_id=room.id,
            index=1,
            body="refused",
            status=EventStatus.BLOCKED,
            blocked_by="spam",
            parent_event_id=root.id,
        )
        reply = make_event(room_id=room.id, index=2, body="2", parent_event_id=root.id)
        for event in (root, refused, reply):
            await contract_store.add_event(event)

        assert [e.index for e in await contract_store.list_events(room.id)] == [0, 2]
        assert [e.index for e in await contract_store.get_timeline(room.id)] == [0, 2]
        # The filter runs ahead of the page cut: a page of two received rows is full.
        page = await contract_store.list_events(room.id, before_index=3, limit=2)
        assert [e.index for e in page] == [0, 2]

        everything = await contract_store.list_events(
            room.id, event_filter=EventFilter(include_blocked=True)
        )
        assert [e.index for e in everything] == [0, 1, 2]
        by_id = await contract_store.get_event(refused.id)
        assert by_id is not None and by_id.status == EventStatus.BLOCKED
        assert [e.index for e in await contract_store.get_conversation(room.id)] == [0, 1, 2]
        # The thread affordance and the thread list agree: one received reply.
        thread = await contract_store.list_events(
            room.id, event_filter=EventFilter(parent_event_id=root.id)
        )
        assert [e.index for e in thread] == [2]
        summaries = await contract_store.get_thread_summaries(room.id, [root.id])
        assert summaries[root.id].reply_count == 1
        assert summaries[root.id].last_reply_at == reply.created_at
    finally:
        await contract_store.delete_room(room.id)


async def test_a_filtered_count_is_the_page_it_stands_for(
    contract_store: ConversationStore,
) -> None:
    """RFC §14.1: without a filter the count is the timeline's size, refused
    rows included (a refused row consumed an index, §8.3); with a filter it
    counts exactly what ``list_events`` would serve under that filter, with no
    page, the received-rows default lifted by ``include_blocked``."""
    room = await contract_store.create_room(Room(id=uuid4().hex))
    t0 = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
    try:
        rows = [
            make_event(
                room_id=room.id,
                index=0,
                body="q1",
                channel_id="ws",
                channel_type=ChannelType.WEBSOCKET,
                created_at=t0,
            ),
            make_event(
                room_id=room.id,
                index=1,
                body="a1",
                channel_id="agent",
                channel_type=ChannelType.AI,
                created_at=t0 + timedelta(minutes=1),
            ),
            make_event(
                room_id=room.id,
                index=2,
                body="refused",
                channel_id="agent",
                channel_type=ChannelType.AI,
                status=EventStatus.BLOCKED,
                blocked_by="budget",
                created_at=t0 + timedelta(minutes=2),
            ),
            make_event(
                room_id=room.id,
                index=3,
                body="a2",
                channel_id="agent",
                channel_type=ChannelType.AI,
                created_at=t0 + timedelta(minutes=3),
            ),
            make_event(
                room_id=room.id,
                index=4,
                body="joined",
                type=EventType.SYSTEM,
                created_at=t0 + timedelta(minutes=4),
            ),
            make_event(
                room_id=room.id,
                index=5,
                body="psst",
                channel_id="ws",
                channel_type=ChannelType.WEBSOCKET,
                participant_id="alice",
                visibility="private",
                created_at=t0 + timedelta(minutes=5),
            ),
        ]
        for event in rows:
            await contract_store.add_event(event)
        ai_turns = EventFilter(event_types=[EventType.MESSAGE], source_channel_type=ChannelType.AI)

        assert await contract_store.get_event_count(room.id) == 6
        assert await contract_store.get_event_count(room.id, EventFilter()) == 5
        assert await contract_store.get_event_count(room.id, ai_turns) == 2
        assert (
            await contract_store.get_event_count(
                room.id,
                EventFilter(
                    event_types=[EventType.MESSAGE],
                    source_channel_type=ChannelType.AI,
                    include_blocked=True,
                ),
            )
            == 3
        )
        assert (
            await contract_store.get_event_count(
                room.id,
                EventFilter(
                    event_types=[EventType.MESSAGE],
                    source_channel_type=ChannelType.AI,
                    before_time=t0 + timedelta(minutes=3),
                ),
            )
            == 1
        )
        # The two received messages written after the first minute: "a2" and
        # the private one. The refused row sits between them and is not counted.
        assert (
            await contract_store.get_event_count(
                room.id,
                EventFilter(
                    event_types=[EventType.MESSAGE],
                    after_time=t0 + timedelta(minutes=1),
                ),
            )
            == 2
        )
        assert (
            await contract_store.get_event_count(room.id, EventFilter(participant_id="alice")) == 1
        )
        assert (
            await contract_store.get_event_count(room.id, EventFilter(visibility="private")) == 1
        )
        # The count is the page it stands for, on every criterion.
        for criteria in (
            ai_turns,
            EventFilter(participant_id="alice"),
            EventFilter(visibility="private"),
            EventFilter(after_time=t0 + timedelta(minutes=1)),
            EventFilter(include_blocked=True),
        ):
            page = await contract_store.list_events(room.id, limit=1000, event_filter=criteria)
            assert await contract_store.get_event_count(room.id, criteria) == len(page)
    finally:
        await contract_store.delete_room(room.id)


async def test_pages_are_rendered_by_index_not_by_clock_or_write_order(
    contract_store: ConversationStore,
) -> None:
    """RFC §14.1: the page is ascending by ``index`` whatever the order the rows
    were written or stamped in. ``created_at`` is stamped when an event is built
    and the index reserved at commit, so a concurrent commit or a backfill makes
    the two disagree; a backend sorting by clock or by insertion renders the
    same room differently from the others."""
    room = await contract_store.create_room(Room(id=uuid4().hex))
    try:
        stamped = datetime.now(UTC)
        # Written 1, 0, 2 with the clock running backwards along the index.
        for i in (1, 0, 2):
            await contract_store.add_event(
                make_event(
                    room_id=room.id,
                    index=i,
                    body=str(i),
                    created_at=stamped - timedelta(seconds=i),
                )
            )
        head = await contract_store.list_events(room.id)
        tail = await contract_store.list_events(room.id, limit=2, newest_first=True)
        forward = await contract_store.list_events(room.id, after_index=0)
        assert [e.index for e in head] == [0, 1, 2]
        assert [e.index for e in tail] == [1, 2]
        assert [e.index for e in forward] == [1, 2]
    finally:
        await contract_store.delete_room(room.id)


class _StoreWithoutTheLookup(InMemoryStore):
    """A store that does not override ``find_room_id_by_binding``: the base fallback."""

    find_room_id_by_binding = ConversationStore.find_room_id_by_binding


@pytest.mark.parametrize("fallback", [False, True])
async def test_a_binding_names_its_sender_for_its_own_channel_only(
    contract_store: ConversationStore, fallback: bool
) -> None:
    """RFC §10.4 step 1: the newest room whose binding of this channel names
    the sender, the room id breaking a tie, the same in every store."""
    store = _StoreWithoutTheLookup() if fallback else contract_store
    if fallback and not isinstance(contract_store, InMemoryStore):
        pytest.skip("the fallback runs over the in-memory store")
    channel, other = f"sms-{uuid4().hex}", f"sms-{uuid4().hex}"
    t0 = datetime(2026, 10, 8, 12, tzinfo=UTC)
    tag = uuid4().hex
    rooms = [
        (Room(id=f"a-{tag}", created_at=t0), channel, "alice"),
        (Room(id=f"b-{tag}", created_at=t0 + timedelta(seconds=1)), channel, "alice"),
        (Room(id=f"c-{tag}", created_at=t0 + timedelta(seconds=1)), channel, "alice"),
        (Room(id=f"d-{tag}", created_at=t0 + timedelta(seconds=2)), channel, "bob"),
        (Room(id=f"e-{tag}", created_at=t0 + timedelta(seconds=3)), other, "alice"),
        (
            Room(id=f"f-{tag}", created_at=t0 + timedelta(seconds=4), status=RoomStatus.CLOSED),
            channel,
            "alice",
        ),
    ]
    for room, channel_id, sender in rooms:
        await store.create_room(room)
        await store.add_binding(
            ChannelBinding(
                room_id=room.id,
                channel_id=channel_id,
                channel_type=ChannelType.SMS,
                participant_id=sender,
            )
        )
    active = str(RoomStatus.ACTIVE)
    try:
        assert await store.find_room_id_by_binding(channel, "alice", status=active) == f"c-{tag}"
        assert await store.find_room_id_by_binding(channel, "alice") == f"f-{tag}"
        assert await store.find_room_id_by_binding(channel, "bob", status=active) == f"d-{tag}"
        assert await store.find_room_id_by_binding(other, "bob", status=active) is None
        assert await store.find_room_id_by_binding(channel, "carol") is None
    finally:
        for room, _channel_id, _sender in rooms:
            await store.delete_room(room.id)


async def test_step_three_gives_the_same_answer_in_every_store(
    contract_store: ConversationStore,
) -> None:
    """RFC §10.4: what the one room bound to a channel received decides who it
    admits, whatever the backend."""
    channel = f"sms-{uuid4().hex}"
    room = await contract_store.create_room(Room(id=uuid4().hex))
    binding = ChannelBinding(room_id=room.id, channel_id=channel, channel_type=ChannelType.SMS)
    await contract_store.add_binding(binding)
    router = DefaultInboundRoomRouter(contract_store)

    async def routes(sender: str) -> str | None:
        return await router.route(channel, ChannelType.SMS, participant_id=sender)

    try:
        assert await routes("bob") == room.id
        await contract_store.commit_event(
            room.id,
            make_event(room_id=room.id, channel_id=channel, participant_id="alice"),
        )
        assert await routes("alice") == room.id
        assert await routes("bob") is None
        await contract_store.update_binding(binding.model_copy(update={"group": True}))
        assert await routes("bob") == room.id
    finally:
        await contract_store.delete_room(room.id)


class _StoreWithoutTheParticipantLookup(InMemoryStore):
    """A store that does not override ``find_room_id_by_participant``."""

    find_room_id_by_participant = ConversationStore.find_room_id_by_participant


@pytest.mark.parametrize("fallback", [False, True])
async def test_a_participant_record_finds_its_room_never_a_binding(
    contract_store: ConversationStore, fallback: bool
) -> None:
    """RFC §10.4 step 1, second half: the newest room where the sender is a
    participant, the room id breaking a tie; a binding naming them is not one."""
    store = _StoreWithoutTheParticipantLookup() if fallback else contract_store
    if fallback and not isinstance(contract_store, InMemoryStore):
        pytest.skip("the fallback runs over the in-memory store")
    t0 = datetime(2026, 10, 8, 12, tzinfo=UTC)
    tag = uuid4().hex
    who = f"alice-{tag}"
    rooms = [
        Room(id=f"a-{tag}", created_at=t0),
        Room(id=f"b-{tag}", created_at=t0 + timedelta(seconds=1)),
        Room(id=f"c-{tag}", created_at=t0 + timedelta(seconds=1)),
        Room(id=f"d-{tag}", created_at=t0 + timedelta(seconds=2), status=RoomStatus.CLOSED),
        Room(id=f"e-{tag}", created_at=t0 + timedelta(seconds=3)),
    ]
    for room in rooms:
        await store.create_room(room)
    for room in rooms[:4]:
        await store.add_participant(Participant(id=who, room_id=room.id, channel_id="sms"))
    await store.add_binding(
        ChannelBinding(
            room_id=f"e-{tag}", channel_id="sms", channel_type=ChannelType.SMS, participant_id=who
        )
    )
    active = str(RoomStatus.ACTIVE)
    try:
        assert await store.find_room_id_by_participant(who, status=active) == f"c-{tag}"
        assert await store.find_room_id_by_participant(who) == f"d-{tag}"
        assert await store.find_room_id_by_participant(f"bob-{tag}") is None
    finally:
        for room in rooms:
            await store.delete_room(room.id)


class _StoreWithoutTheRoutingLookups(InMemoryStore):
    """A store that overrides none of the routing lookups: the base fallbacks,
    over a single-room ``find_room_id_by_channel`` of its own."""

    find_room_id_by_binding = ConversationStore.find_room_id_by_binding
    find_room_id_by_participant = ConversationStore.find_room_id_by_participant
    find_room_ids_by_channel = ConversationStore.find_room_ids_by_channel

    async def find_room_id_by_channel(
        self, channel_id: str, status: str | None = None
    ) -> str | None:
        found = await InMemoryStore.find_room_ids_by_channel(self, channel_id, status, limit=1)
        return found[0] if found else None


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("step", ["binding", "participant", "one-room"])
async def test_the_default_router_reads_the_organizations_rooms_only(
    contract_store: ConversationStore, fallback: bool, step: str
) -> None:
    """RFC §10.4: the default strategy searches the caller's organization
    only, every room when the call names none. A newer room of another
    tenant neither wins over the organization's own nor hides it, and does
    not make the channel's one room of the organization ambiguous."""
    store = _StoreWithoutTheRoutingLookups() if fallback else contract_store
    if fallback and not isinstance(contract_store, InMemoryStore):
        pytest.skip("the fallback runs over the in-memory store")
    channel, tag = f"sms-{uuid4().hex}", uuid4().hex
    sender = f"alice-{tag}"
    t0 = datetime(2026, 10, 8, 12, tzinfo=UTC)
    rooms = [
        Room(id=f"acme-{tag}", organization_id="acme", created_at=t0),
        Room(id=f"globex-{tag}", organization_id="globex", created_at=t0 + timedelta(seconds=1)),
    ]
    for room in rooms:
        await store.create_room(room)
        await store.add_binding(
            ChannelBinding(
                room_id=room.id,
                channel_id=channel,
                channel_type=ChannelType.SMS,
                participant_id=sender if step == "binding" else None,
            )
        )
        if step == "participant":
            await store.add_participant(
                Participant(id=sender, room_id=room.id, channel_id=channel)
            )
    router = DefaultInboundRoomRouter(store)

    async def routes(organization_id: str | None) -> str | None:
        return await router.route(
            channel, ChannelType.SMS, participant_id=sender, organization_id=organization_id
        )

    try:
        assert await routes("acme") == f"acme-{tag}"
        assert await routes("globex") == f"globex-{tag}"
        assert await routes("initech") is None
        if not fallback:
            # Unscoped, both rooms are candidates: the newest is the sender's
            # (step 1), and two rooms bound to the channel make step 3
            # ambiguous, which the unscoped fallback cannot tell (it reads
            # one room).
            assert await routes(None) == (None if step == "one-room" else f"globex-{tag}")
    finally:
        for room in rooms:
            await store.delete_room(room.id)


def _edit_of(target: RoomEvent, body: str) -> tuple[RoomEvent, RoomEvent]:
    """A DELIVERED edit of *target*, and *target* as its commit writes it."""
    new = TextContent(body=body)
    edit = make_event(room_id=target.room_id, status=EventStatus.DELIVERED).model_copy(
        update={
            "type": EventType.EDIT,
            "content": EditContent(target_event_id=target.id, new_content=new),
        }
    )
    edited = target.model_copy(update={"content": new, "metadata": {"edited": True}})
    return edit, edited


async def test_a_change_commits_with_its_target_or_not_at_all(
    contract_store: ConversationStore,
) -> None:
    """RFC §14.3: a DELIVERED edit, the room's counters and its target's
    update are one transaction, and a target the room no longer holds, or
    another room's, fails the whole commit."""
    room = await contract_store.create_room(Room(id=uuid4().hex))
    other = await contract_store.create_room(Room(id=uuid4().hex))
    try:
        target = await contract_store.commit_event(room.id, make_event(room_id=room.id))
        edit, edited = _edit_of(target, "v2")

        committed = await contract_store.commit_event(room.id, edit, target=edited)

        stored = await contract_store.get_event(target.id)
        assert stored is not None
        assert (stored.content, stored.metadata) == (TextContent(body="v2"), {"edited": True})
        assert await contract_store.get_event(committed.id) is not None

        foreign = await contract_store.commit_event(other.id, make_event(room_id=other.id))
        for gone in (make_event(room_id=room.id), foreign.model_copy(update={"room_id": room.id})):
            orphan, orphan_target = _edit_of(gone, "v3")
            with pytest.raises(EventNotFoundError):
                await contract_store.commit_event(room.id, orphan, target=orphan_target)
            assert await contract_store.get_event(orphan.id) is None
        kept = await contract_store.get_event(foreign.id)
        assert kept is not None and kept.content == foreign.content
        after = await contract_store.get_room(room.id)
        assert after is not None
        assert (after.event_count, after.latest_index) == (2, committed.index)
    finally:
        await contract_store.delete_room(room.id)
        await contract_store.delete_room(other.id)


def _delete_of(target: RoomEvent) -> RoomEvent:
    """A DELIVERED delete of *target*."""
    return make_event(room_id=target.room_id, status=EventStatus.DELIVERED).model_copy(
        update={"type": EventType.DELETE, "content": DeleteContent(target_event_id=target.id)}
    )


@pytest.mark.parametrize("cascade", [True, False])
async def test_a_hard_delete_takes_every_removed_event_s_changes(
    contract_store: ConversationStore, cascade: bool
) -> None:
    """RFC §10.3, §14.1: a hard delete removes its event, its thread replies
    unless told not to, and the edits and deletes of every event it removes
    (an edit holds its target's corrected text), and returns them as stored:
    the root, its replies, then the changes, in index order."""
    room = await contract_store.create_room(Room(id=uuid4().hex))
    try:
        store = contract_store
        root = await store.commit_event(room.id, make_event(room_id=room.id, body="root"))
        reply = await store.commit_event(
            room.id, make_event(room_id=room.id, body="reply", parent_event_id=root.id)
        )
        other = await store.commit_event(room.id, make_event(room_id=room.id, body="other"))
        edit, edited = _edit_of(root, "corrected")
        edit = await store.commit_event(room.id, edit, target=edited)
        unsaid = await store.commit_event(room.id, _delete_of(reply))
        kept_edit, kept_edited = _edit_of(other, "other, corrected")
        kept_edit = await store.commit_event(room.id, kept_edit, target=kept_edited)
        as_stored = await store.get_event(root.id)

        removed = await store.delete_event(room.id, root.id, cascade_replies=cascade)

        expected = [root, reply, edit, unsaid] if cascade else [root, edit]
        assert [e.id for e in removed] == [e.id for e in expected]
        assert removed[0] == as_stored
        left = {e.id for e in await store.list_events(room.id)}
        assert left == {other.id, kept_edit.id} | (set() if cascade else {reply.id, unsaid.id})
        assert await store.delete_event(room.id, root.id) == []
        assert await store.delete_event(uuid4().hex, other.id) == []
    finally:
        await contract_store.delete_room(room.id)
