"""A WebSocket connection receives its rooms, and only its rooms.

The registry used to be a flat ``{connection_id: send_fn}``, with no room
dimension anywhere in the API — so ``deliver()`` had nothing to filter on and
sent every room's events to every socket the channel held. A channel shared
across conversations therefore leaked them into each other, durably: the
client saw them, and so did anything reading that socket.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

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


async def _acme_room_on_a_socket_channel() -> tuple[RoomKit, WebSocketChannel]:
    kit = RoomKit()
    ws = WebSocketChannel("ws")
    kit.register_channel(ws)
    kit.register_channel(SimpleChannel("sms1"))
    await kit.create_room(room_id="acme-room", organization_id="acme")
    await kit.attach_channel("acme-room", "ws")
    await kit.attach_channel("acme-room", "sms1")
    return kit, ws


SubscribeDoor = Callable[[RoomKit, WebSocketChannel, str, SendFn, str | None], None]


def _by_register(
    kit: RoomKit, ws: WebSocketChannel, conn: str, send: SendFn, org: str | None
) -> None:
    ws.register_connection(conn, send, room_id="acme-room", organization_id=org)


def _by_subscribe(
    kit: RoomKit, ws: WebSocketChannel, conn: str, send: SendFn, org: str | None
) -> None:
    ws.register_connection(conn, send, room_id="lobby")
    ws.subscribe(conn, "acme-room", organization_id=org)


def _by_kit_subscribe(
    kit: RoomKit, ws: WebSocketChannel, conn: str, send: SendFn, org: str | None
) -> None:
    ws.register_connection(conn, send, room_id="lobby")
    kit.subscribe_websocket("ws", conn, "acme-room", organization_id=org)


_SUBSCRIBE_DOORS: dict[str, SubscribeDoor] = {
    "register_connection": _by_register,
    "subscribe": _by_subscribe,
    "subscribe_websocket": _by_kit_subscribe,
}


class TestOrganizationScope:
    """A connection subscribed for an organization receives only the rooms
    of that organization (RFC §17.2, RMK-491): the subscription is
    synchronous and reads no room, so the room is matched at delivery."""

    @pytest.mark.parametrize("door", sorted(_SUBSCRIBE_DOORS))
    async def test_a_connection_of_another_organization_receives_nothing(self, door: str) -> None:
        kit, ws = await _acme_room_on_a_socket_channel()
        seen: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            seen.append(conn_id)

        for conn, org in (("acme-socket", "acme"), ("globex-socket", "globex"), ("any", None)):
            _SUBSCRIBE_DOORS[door](kit, ws, conn, send, org)

        await kit.process_inbound(
            InboundMessage(
                channel_id="sms1", sender_id="+15550001", content=TextContent(body="hi")
            ),
            room_id="acme-room",
        )

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

    @pytest.mark.parametrize("door", ["unsubscribe", "unsubscribe_websocket"])
    async def test_another_organization_does_not_unsubscribe_a_connection(self, door: str) -> None:
        kit, ws = await _acme_room_on_a_socket_channel()

        async def send(conn_id: str, event: RoomEvent) -> None:
            return None

        ws.register_connection("acme-socket", send, room_id="acme-room", organization_id="acme")

        def unsubscribe(org: str | None) -> None:
            if door == "unsubscribe":
                ws.unsubscribe("acme-socket", "acme-room", organization_id=org)
            else:
                kit.unsubscribe_websocket("ws", "acme-socket", "acme-room", organization_id=org)

        unsubscribe("globex")
        assert ws.rooms_for("acme-socket") == {"acme-room"}
        unsubscribe("acme")
        assert ws.rooms_for("acme-socket") == set()
        await kit.close()
