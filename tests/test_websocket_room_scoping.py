"""A WebSocket connection receives its rooms, and only its rooms.

The registry used to be a flat ``{connection_id: send_fn}``, with no room
dimension anywhere in the API — so ``deliver()`` had nothing to filter on and
sent every room's events to every socket the channel held. A channel shared
across conversations therefore leaked them into each other, durably: the
client saw them, and so did anything reading that socket.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.websocket import SendFn, WebSocketChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from tests.test_framework import SimpleChannel


def _binding(room_id: str, channel_id: str = "ws") -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id, room_id=room_id, channel_type=ChannelType.WEBSOCKET
    )


def _context(room_id: str) -> RoomContext:
    return RoomContext(room=Room(id=room_id))


def _event(room_id: str, body: str) -> RoomEvent:
    return RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id="other", channel_type=ChannelType.WEBSOCKET),
        content=TextContent(body=body),
    )


class TestDeliveryIsScopedToTheRoom:
    async def test_a_room_reaches_only_its_own_connections(self) -> None:
        ws = WebSocketChannel("ws")
        seen: list[tuple[str, str]] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            seen.append((conn_id, event.content.body))

        ws.register_connection("alice", send, room_id="room-a")
        ws.register_connection("bob", send, room_id="room-b")

        await ws.deliver(_event("room-a", "secret"), _binding("room-a"), _context("room-a"))

        assert seen == [("alice", "secret")]

    async def test_several_connections_in_one_room_all_receive(self) -> None:
        ws = WebSocketChannel("ws")
        seen: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            seen.append(conn_id)

        ws.register_connection("a1", send, room_id="room-a")
        ws.register_connection("a2", send, room_id="room-a")
        ws.register_connection("b1", send, room_id="room-b")

        await ws.deliver(_event("room-a", "hi"), _binding("room-a"), _context("room-a"))

        assert sorted(seen) == ["a1", "a2"]

    async def test_a_room_with_no_connections_delivers_to_nobody(self) -> None:
        ws = WebSocketChannel("ws")
        seen: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            seen.append(conn_id)

        ws.register_connection("alice", send, room_id="room-a")

        await ws.deliver(_event("room-z", "hi"), _binding("room-z"), _context("room-z"))

        assert seen == []


class TestSubscription:
    async def test_one_socket_can_follow_several_rooms(self) -> None:
        ws = WebSocketChannel("ws")
        seen: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            seen.append(event.room_id)

        ws.register_connection("alice", send, room_id="room-a")
        ws.subscribe("alice", "room-b")

        await ws.deliver(_event("room-a", "x"), _binding("room-a"), _context("room-a"))
        await ws.deliver(_event("room-b", "y"), _binding("room-b"), _context("room-b"))

        assert seen == ["room-a", "room-b"]

    async def test_unsubscribing_stops_delivery(self) -> None:
        ws = WebSocketChannel("ws")
        seen: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            seen.append(event.room_id)

        ws.register_connection("alice", send, room_id="room-a")
        ws.subscribe("alice", "room-b")
        ws.unsubscribe("alice", "room-b")

        await ws.deliver(_event("room-b", "y"), _binding("room-b"), _context("room-b"))

        assert seen == []
        assert ws.rooms_for("alice") == {"room-a"}

    async def test_unregistering_drops_every_subscription(self) -> None:
        ws = WebSocketChannel("ws")

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        ws.register_connection("alice", send, room_id="room-a")
        ws.subscribe("alice", "room-b")
        ws.unregister_connection("alice")

        assert ws.rooms_for("alice") == set()
        assert ws._room_connections == {}

    async def test_room_id_is_required(self) -> None:
        """The parameter exists so a connection can never be unplaceable."""
        ws = WebSocketChannel("ws")

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        with pytest.raises(TypeError):
            ws.register_connection("alice", send)  # ty: ignore[missing-argument]


class TestStreamingIsScopedToo:
    async def test_streaming_capability_is_answered_per_room(self) -> None:
        ws = WebSocketChannel("ws")

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        async def stream_send(conn_id: str, msg: object) -> None:
            pass

        ws.register_connection("alice", send, room_id="room-a", stream_send_fn=stream_send)
        ws.register_connection("bob", send, room_id="room-b")

        assert ws.supports_streaming_delivery_for("room-a") is True
        assert ws.supports_streaming_delivery_for("room-b") is False
        # The channel-wide answer stays true — some client somewhere streams.
        assert ws.supports_streaming_delivery is True

    async def test_stream_messages_stay_in_their_room(self) -> None:
        ws = WebSocketChannel("ws")
        streamed: list[tuple[str, str]] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        async def stream_send(conn_id: str, msg: object) -> None:
            streamed.append((conn_id, msg.type))

        ws.register_connection("alice", send, room_id="room-a", stream_send_fn=stream_send)
        ws.register_connection("bob", send, room_id="room-b", stream_send_fn=stream_send)

        async def text_stream():
            yield "hello"

        await ws.deliver_stream(
            text_stream(), _event("room-a", ""), _binding("room-a"), _context("room-a")
        )

        assert {conn_id for conn_id, _ in streamed} == {"alice"}


class TestFanOutIsBounded:
    """A socket that hangs must not take the room with it.

    Broadcast runs under the room lock and is unbounded by design (RFC §10.1),
    so an unbounded send does not merely delay other clients — it stops the
    room from accepting anything at all.
    """

    async def test_a_hanging_connection_does_not_block_the_others(self) -> None:
        ws = WebSocketChannel("ws", send_timeout=0.05)
        delivered: list[str] = []

        async def hang(conn_id: str, event: RoomEvent) -> None:
            await asyncio.sleep(30)

        async def fast(conn_id: str, event: RoomEvent) -> None:
            delivered.append(conn_id)

        ws.register_connection("zombie", hang, room_id="room-a")
        ws.register_connection("alice", fast, room_id="room-a")

        # Back once the hung socket's send_timeout ran out, never at its pace.
        await asyncio.wait_for(
            ws.deliver(_event("room-a", "hi"), _binding("room-a"), _context("room-a")), 1.0
        )

        assert delivered == ["alice"]

    async def test_a_hanging_connection_is_eventually_evicted(self) -> None:
        """A half-open socket never raises, so only a timeout can retire it."""
        ws = WebSocketChannel("ws", send_timeout=0.01)

        async def hang(conn_id: str, event: RoomEvent) -> None:
            await asyncio.sleep(30)

        ws.register_connection("zombie", hang, room_id="room-a")

        for _ in range(WebSocketChannel._MAX_CONSECUTIVE_ERRORS):
            await ws.deliver(_event("room-a", "x"), _binding("room-a"), _context("room-a"))

        assert "zombie" not in ws._connections

    async def test_a_recovering_connection_keeps_its_place(self) -> None:
        ws = WebSocketChannel("ws", send_timeout=0.05)
        calls = {"n": 0}

        async def flaky(conn_id: str, event: RoomEvent) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                await asyncio.sleep(30)

        ws.register_connection("flaky", flaky, room_id="room-a")

        await ws.deliver(_event("room-a", "1"), _binding("room-a"), _context("room-a"))
        await ws.deliver(_event("room-a", "2"), _binding("room-a"), _context("room-a"))

        assert "flaky" in ws._connections
        assert ws._error_counts.get("flaky", 0) == 0

    async def test_slow_clients_do_not_add_up(self) -> None:
        """Concurrent, not sequential: each socket's send is under way while the
        others start, so none waits for another to finish."""
        ws = WebSocketChannel("ws", send_timeout=5.0)
        started: list[str] = []
        delivered: list[str] = []
        all_started = asyncio.Event()

        async def slow(conn_id: str, event: RoomEvent) -> None:
            started.append(conn_id)
            if len(started) == 5:
                all_started.set()
            # A sequential fan-out never starts the others while this one waits.
            await asyncio.wait_for(all_started.wait(), 1.0)
            delivered.append(conn_id)

        for i in range(5):
            ws.register_connection(f"c{i}", slow, room_id="room-a")

        await ws.deliver(_event("room-a", "hi"), _binding("room-a"), _context("room-a"))

        assert sorted(delivered) == [f"c{i}" for i in range(5)], "fan-out was sequential"


async def _two_rooms_on_a_socket_channel() -> tuple[RoomKit, WebSocketChannel]:
    kit = RoomKit()
    ws = WebSocketChannel("ws")
    kit.register_channel(ws)
    kit.register_channel(SimpleChannel("sms1"))
    for room_id, org in (("acme-room", "acme"), ("globex-room", "globex")):
        await kit.create_room(room_id=room_id, organization_id=org)
        await kit.attach_channel(room_id, "ws")
        await kit.attach_channel(room_id, "sms1")
    return kit, ws


async def _say_in(kit: RoomKit, room_id: str) -> None:
    message = InboundMessage(
        channel_id="sms1", sender_id="+15550001", content=TextContent(body="hi")
    )
    await kit.process_inbound(message, room_id=room_id)


class _Socket:
    """A client socket: what it was sent."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def send(self, conn_id: str, event: RoomEvent) -> None:
        self.seen.append(event.room_id)


SubscribeDoor = Callable[[RoomKit, WebSocketChannel, str, SendFn, str | None], None]


def _by_register(
    kit: RoomKit, ws: WebSocketChannel, conn: str, send: SendFn, org: str | None
) -> None:
    ws.register_connection(conn, send, room_id="acme-room", organization_id=org)


def _by_subscribe(
    kit: RoomKit, ws: WebSocketChannel, conn: str, send: SendFn, org: str | None
) -> None:
    ws.register_connection(conn, send, room_id="lobby", organization_id=org)
    ws.subscribe(conn, "acme-room", organization_id=org)


def _by_kit_subscribe(
    kit: RoomKit, ws: WebSocketChannel, conn: str, send: SendFn, org: str | None
) -> None:
    ws.register_connection(conn, send, room_id="lobby", organization_id=org)
    kit.subscribe_websocket("ws", conn, "acme-room", organization_id=org)


_SUBSCRIBE_DOORS: dict[str, SubscribeDoor] = {
    "register_connection": _by_register,
    "subscribe": _by_subscribe,
    "subscribe_websocket": _by_kit_subscribe,
}


async def _taken_over(door: str, kit: RoomKit, ws: WebSocketChannel, intruder: _Socket) -> None:
    """Globex's host acting on the connection acme registered."""
    if door == "register_connection":
        ws.register_connection(
            "acme-socket", intruder.send, room_id="acme-room", organization_id="globex"
        )
    elif door == "connect_websocket":
        await kit.connect_websocket(
            "ws", "acme-socket", intruder.send, room_id="globex-room", organization_id="globex"
        )
    elif door == "subscribe":
        ws.subscribe("acme-socket", "acme-room", organization_id="globex")
    elif door == "subscribe_websocket":
        kit.subscribe_websocket("ws", "acme-socket", "acme-room", organization_id="globex")
    elif door == "unsubscribe":
        ws.unsubscribe("acme-socket", "acme-room", organization_id="globex")
    elif door == "unsubscribe_websocket":
        kit.unsubscribe_websocket("ws", "acme-socket", "acme-room", organization_id="globex")
    elif door == "unregister_connection":
        ws.unregister_connection("acme-socket", organization_id="globex")
    else:
        await kit.disconnect_websocket("ws", "acme-socket", organization_id="globex")


_TAKEOVER_DOORS = [
    "connect_websocket",
    "disconnect_websocket",
    "register_connection",
    "unregister_connection",
    "subscribe",
    "subscribe_websocket",
    "unsubscribe",
    "unsubscribe_websocket",
]


class TestOrganizationScope:
    """A connection registered for an organization receives only the rooms
    of that organization, and only that organization subscribes or
    unsubscribes it (RFC §17.2, RMK-491). The registry is synchronous and
    reads no room, so the room is matched at delivery."""

    @pytest.mark.parametrize("door", sorted(_SUBSCRIBE_DOORS))
    async def test_a_connection_of_another_organization_receives_nothing(self, door: str) -> None:
        kit, ws = await _two_rooms_on_a_socket_channel()
        seen: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            seen.append(conn_id)

        for conn, org in (("acme-socket", "acme"), ("globex-socket", "globex"), ("any", None)):
            _SUBSCRIBE_DOORS[door](kit, ws, conn, send, org)

        await _say_in(kit, "acme-room")

        assert sorted(seen) == ["acme-socket", "any"]
        await kit.close()

    async def test_a_stream_reaches_no_connection_of_another_organization(self) -> None:
        ws = WebSocketChannel("ws")
        streamed: list[str] = []
        sent: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            sent.append(conn_id)

        async def stream_send(conn_id: str, msg: object) -> None:
            streamed.append(conn_id)

        for conn, org in (("acme-live", "acme"), ("globex-live", "globex")):
            ws.register_connection(
                conn, send, room_id="room-a", stream_send_fn=stream_send, organization_id=org
            )
        for conn, org in (("acme-final", "acme"), ("globex-final", "globex")):
            ws.register_connection(conn, send, room_id="room-a", organization_id=org)

        async def text_stream() -> AsyncIterator[str]:
            yield "hello"

        context = RoomContext(room=Room(id="room-a", organization_id="acme"))
        await ws.deliver_stream(text_stream(), _event("room-a", ""), _binding("room-a"), context)

        assert set(streamed) == {"acme-live"}
        assert sent == ["acme-final"]

    @pytest.mark.parametrize("door", _TAKEOVER_DOORS)
    async def test_another_organization_cannot_take_over_a_connection(self, door: str) -> None:
        """Refused as a connection that does not exist, nothing changed: the
        socket keeps its sender, its organization and its rooms."""
        kit, ws = await _two_rooms_on_a_socket_channel()
        acme, intruder = _Socket(), _Socket()
        ws.register_connection(
            "acme-socket", acme.send, room_id="acme-room", organization_id="acme"
        )

        with pytest.raises(ValueError, match="No connection acme-socket for organization globex"):
            await _taken_over(door, kit, ws, intruder)

        await _say_in(kit, "acme-room")
        await _say_in(kit, "globex-room")
        assert (acme.seen, intruder.seen) == (["acme-room"], [])
        assert ws.rooms_for("acme-socket") == {"acme-room"}
        await kit.close()

    @pytest.mark.parametrize("kit_door", [False, True], ids=["channel", "kit"])
    async def test_an_unscoped_subscription_never_widens_a_scoped_connection(
        self, kit_door: bool
    ) -> None:
        kit, ws = await _two_rooms_on_a_socket_channel()
        acme = _Socket()
        ws.register_connection(
            "acme-socket", acme.send, room_id="acme-room", organization_id="acme"
        )

        if kit_door:
            kit.subscribe_websocket("ws", "acme-socket", "globex-room")
            kit.subscribe_websocket("ws", "acme-socket", "acme-room")
        else:
            ws.subscribe("acme-socket", "globex-room")
            ws.subscribe("acme-socket", "acme-room")
        await _say_in(kit, "globex-room")
        await _say_in(kit, "acme-room")

        assert acme.seen == ["acme-room"]
        await kit.close()

    @pytest.mark.parametrize("kit_door", [False, True], ids=["channel", "kit"])
    async def test_the_connections_own_organization_subscribes_and_unsubscribes_it(
        self, kit_door: bool
    ) -> None:
        kit, ws = await _two_rooms_on_a_socket_channel()
        acme = _Socket()
        ws.register_connection("acme-socket", acme.send, room_id="lobby", organization_id="acme")

        if kit_door:
            kit.subscribe_websocket("ws", "acme-socket", "acme-room", organization_id="acme")
            await _say_in(kit, "acme-room")
            kit.unsubscribe_websocket("ws", "acme-socket", "acme-room", organization_id="acme")
        else:
            ws.subscribe("acme-socket", "acme-room", organization_id="acme")
            await _say_in(kit, "acme-room")
            ws.unsubscribe("acme-socket", "acme-room", organization_id="acme")
        await _say_in(kit, "acme-room")

        assert acme.seen == ["acme-room"]
        assert ws.rooms_for("acme-socket") == {"lobby"}
        await kit.close()

    @pytest.mark.parametrize(
        ("before", "registered_for", "rooms"),
        [
            pytest.param("acme", None, {"lobby"}, id="scoped-then-unscoped"),
            pytest.param("unregistered", "globex", {"lobby"}, id="subscribed-then-scoped"),
            pytest.param("acme", "acme", {"acme-room", "lobby"}, id="same-organization"),
        ],
    )
    async def test_a_registration_that_changes_the_scope_drops_the_old_subscriptions(
        self, before: str, registered_for: str | None, rooms: set[str]
    ) -> None:
        """A new scope on an id still known starts with no subscription: what
        the id followed before is no room of the new registration's."""
        kit, ws = await _two_rooms_on_a_socket_channel()
        old, new = _Socket(), _Socket()
        if before == "unregistered":
            ws.subscribe("c", "acme-room")
        else:
            ws.register_connection("c", old.send, room_id="acme-room", organization_id=before)

        ws.register_connection("c", new.send, room_id="lobby", organization_id=registered_for)
        await _say_in(kit, "acme-room")

        assert ws.rooms_for("c") == rooms
        assert new.seen == (["acme-room"] if "acme-room" in rooms else [])
        await kit.close()

    async def test_an_evicted_connection_is_forgotten_with_its_rooms(self) -> None:
        """As unregister_connection() forgets it: an id registered again after
        the eviction starts with no room and no organization."""
        kit, ws = await _two_rooms_on_a_socket_channel()
        new = _Socket()

        async def gone(conn_id: str, event: RoomEvent) -> None:
            raise ConnectionError("closed")

        ws.register_connection("c", gone, room_id="acme-room", organization_id="acme")
        for _ in range(WebSocketChannel._MAX_CONSECUTIVE_ERRORS):
            await _say_in(kit, "acme-room")
        assert ws.rooms_for("c") == set()

        ws.register_connection("c", new.send, room_id="lobby")
        await _say_in(kit, "acme-room")
        assert new.seen == []
        await kit.close()

    @pytest.mark.parametrize("kit_door", [False, True], ids=["channel", "kit"])
    async def test_the_connections_own_organization_unregisters_it(self, kit_door: bool) -> None:
        kit, ws = await _two_rooms_on_a_socket_channel()
        acme = _Socket()
        ws.register_connection(
            "acme-socket", acme.send, room_id="acme-room", organization_id="acme"
        )

        if kit_door:
            await kit.disconnect_websocket("ws", "acme-socket", organization_id="acme")
        else:
            ws.unregister_connection("acme-socket", organization_id="acme")
        await _say_in(kit, "acme-room")

        assert (acme.seen, ws.connection_count, ws.rooms_for("acme-socket")) == ([], 0, set())
        await kit.close()

    async def test_a_stream_under_way_never_reaches_an_id_registered_again(self) -> None:
        """The stream's connections are the ones registered when it started:
        an id unregistered and registered again for globex meanwhile
        receives nothing of acme's stream, and neither does the socket that
        left (review probe P1)."""
        ws = WebSocketChannel("ws")
        got: dict[str, list[str]] = {"acme": [], "globex": []}

        def stream_to(who: str) -> Callable[[str, Any], Awaitable[None]]:
            async def send(conn_id: str, msg: Any) -> None:
                got[who].append(msg.type)

            return send

        async def plain(conn_id: str, event: RoomEvent) -> None:
            got["acme" if event.room_id == "acme-room" else "globex"].append("event")

        ws.register_connection(
            "c1",
            plain,
            room_id="acme-room",
            stream_send_fn=stream_to("acme"),
            organization_id="acme",
        )
        started = asyncio.Event()
        resume = asyncio.Event()

        async def text() -> AsyncIterator[Any]:
            yield "secret-1 "
            started.set()
            await resume.wait()
            yield "secret-2"
            yield _event("acme-room", "a segment")

        context = RoomContext(room=Room(id="acme-room", organization_id="acme"))
        streaming = asyncio.create_task(
            ws.deliver_stream(text(), _event("acme-room", ""), _binding("acme-room"), context)
        )
        await asyncio.wait_for(started.wait(), 1)
        ws.unregister_connection("c1")
        ws.register_connection(
            "c1",
            plain,
            room_id="globex-room",
            stream_send_fn=stream_to("globex"),
            organization_id="globex",
        )
        resume.set()
        await asyncio.wait_for(streaming, 1)

        assert got == {"acme": ["stream_start", "stream_chunk"], "globex": []}

    async def test_a_room_of_no_organization_reaches_only_unscoped_connections(self) -> None:
        """A room of none is no organization's (RFC §17.2): a connection
        registered for one receives nothing from it, an unscoped one does."""
        ws = WebSocketChannel("ws")
        seen: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            seen.append(conn_id)

        ws.register_connection("acme-socket", send, room_id="room-a", organization_id="acme")
        ws.register_connection("any", send, room_id="room-a")

        await ws.deliver(_event("room-a", "hi"), _binding("room-a"), _context("room-a"))

        assert seen == ["any"]
