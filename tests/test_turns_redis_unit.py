"""The Redis turn backend's parts that need no Redis (RFC §10.6.7, §10.6.8):
its inbox entries, what an executor declares, how an owner counts a beat an
executor makes outside the turn's messages, and the optional dependency."""

from __future__ import annotations

import asyncio
import contextvars
import sys
from typing import Any
from unittest.mock import patch

import pytest

from roomkit import HookTrigger
from roomkit.core.exceptions import TurnLinkError
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.turns import InMemoryTurnBackend, TurnExecutor, TurnTimeouts, _redis_routes
from roomkit.turns._owner_turn import OwnerTurn
from roomkit.turns._redis_beat import LiveExecutors
from roomkit.turns._redis_inbox import Envelope
from roomkit.turns._redis_outbox import Outbox
from roomkit.turns._redis_placement import Placement
from roomkit.turns._redis_presence import Declaration, View, member_of
from roomkit.turns._redis_routes import TAKEN, Routes
from roomkit.turns.owner import registry_turn_client, turn_request
from roomkit.turns.redis import RedisTurnBackend
from roomkit.turns.wire import (
    Accepted,
    ChannelControlEnvelope,
    Ephemeral,
    HostResult,
    OwnerMessageEnvelope,
    RoomCall,
    RoomDetached,
    RoomPublication,
)
from tests.private_module import load_private_copy

QUICK = TurnTimeouts(heartbeat_interval=0.05, heartbeat_timeout=0.2, accept=0.2, cancel=0.2)


def test_an_inbox_entry_reads_back_from_bytes_or_text() -> None:
    sent = Envelope("frame", "node-a", '{"x": 1}', target="turn-1", ref="")
    as_bytes = {key.encode(): value.encode() for key, value in sent.fields().items()}

    assert Envelope.parse(as_bytes) == sent
    assert Envelope.parse(sent.fields()) == sent


def test_a_declaration_takes_its_channels_with_every_room_hook_it_defines() -> None:
    declared = Declaration("node-a", "host-a", frozenset({"meteo"}), frozenset({"audit"}))

    assert declared.takes("meteo", [])
    assert declared.takes("meteo", ["audit"])
    assert not declared.takes("meteo", ["audit", "billing"])
    assert not declared.takes("other", [])
    assert Declaration.load(declared.dump()) == declared


def test_an_executor_given_its_channels_tells_them() -> None:
    listed = TurnExecutor(lambda _: None, hosts={"meteo", "billing"})
    predicate = TurnExecutor(lambda _: None, hosts=lambda cid: cid == "meteo")
    every = TurnExecutor(lambda _: None)

    assert listed.hosted == {"meteo", "billing"} and listed.declares("billing")
    assert predicate.hosted is None and predicate.declares("meteo")
    assert every.hosted is None and every.declares("anything")
    with pytest.raises(TypeError, match="not one id"):
        TurnExecutor(lambda _: None, hosts="meteo")


def test_an_executor_tells_the_names_of_its_room_hooks() -> None:
    executor = TurnExecutor(lambda _: None, hosts={"meteo"})

    @executor.room_hook("audit", HookTrigger.ON_AI_RESPONSE)
    async def audit(event: Any, context: Any) -> None: ...

    assert executor.hooks.defined_room_hooks() == {"audit"}


def test_a_backend_s_timeouts_are_the_kit_s_and_its_own_executor_s() -> None:
    backend = InMemoryTurnBackend()
    backend.timeouts = QUICK

    client = registry_turn_client({}, backend)

    assert client._timeouts is QUICK
    assert client.executor is not None and client.executor._timeouts is QUICK


# -- A beat outside the turn's messages ---------------------------------------------


class _SilentLink:
    """An executor took the turn, then sends nothing: it beats elsewhere."""

    def __init__(self, request: Any, *, beats: bool) -> None:
        self.request = request
        self.taken = True
        self._beats = beats
        self._accepted = False

    async def receive(self) -> Any:
        if not self._accepted:
            self._accepted = True
            return 1, Accepted(executor_id="far")
        await asyncio.Event().wait()

    async def send(self, message: Any) -> None:
        return None

    def release(self) -> None:
        return None

    def beat_heard(self) -> bool:
        return self._beats


def _request() -> Any:
    return turn_request(
        RoomEvent(
            room_id="room",
            source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
            content=TextContent(body="hello"),
        ),
        ChannelBinding(
            channel_id="agent",
            room_id="room",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        ),
        RoomContext(room=Room(id="room")),
    )


async def test_a_turn_whose_executor_beats_outside_its_messages_is_not_lost() -> None:
    turn = OwnerTurn(_SilentLink(_request(), beats=True), QUICK)  # type: ignore[arg-type]
    output = asyncio.ensure_future(turn.output())
    await asyncio.sleep(5 * QUICK.heartbeat_timeout)

    assert not output.done()
    output.cancel()
    await asyncio.gather(output, return_exceptions=True)
    turn.abandon()


async def test_a_turn_whose_executor_stopped_beating_is_lost() -> None:
    turn = OwnerTurn(_SilentLink(_request(), beats=False), QUICK)  # type: ignore[arg-type]

    with pytest.raises(TurnLinkError) as raised:
        await asyncio.wait_for(turn.output(), 5 * QUICK.heartbeat_timeout)

    assert raised.value.reason == "executor_lost"


class _NeverAccepted(_SilentLink):
    """No executor took the turn, though something beats."""

    async def receive(self) -> Any:
        await asyncio.Event().wait()


async def test_a_beat_before_an_executor_took_the_turn_does_not_hold_it() -> None:
    turn = OwnerTurn(_NeverAccepted(_request(), beats=True), QUICK)  # type: ignore[arg-type]

    with pytest.raises(TurnLinkError):
        await asyncio.wait_for(turn.output(), 5 * QUICK.accept)


# -- The optional dependency --------------------------------------------------------


def test_the_backend_without_redis_names_the_extra() -> None:
    fakes: dict[str, Any] = {"redis": None, "redis.asyncio": None}
    with patch.dict(sys.modules, fakes):
        module = load_private_copy("roomkit.turns.redis", fakes)
        with pytest.raises(ImportError, match=r"pip install roomkit\[redis\]"):
            module.RedisTurnBackend()


def test_two_rooms_whose_ids_read_alike_hold_apart() -> None:
    """``support:fr`` in ``acme`` and ``support`` in ``fr:acme`` are two holders."""
    placement = Placement(None, "p", "node", None, loop_back=False)  # type: ignore[arg-type]

    assert placement._holder_key("support:fr", "acme") != placement._holder_key(
        "support", "fr:acme"
    )


def test_an_executor_restarted_elsewhere_is_another_member() -> None:
    assert member_of("node-a", "host-1") != member_of("node-b", "host-1")


def test_a_backend_given_a_client_needs_the_url_its_beat_connects_to() -> None:
    pytest.importorskip("redis.asyncio")

    with pytest.raises(ValueError, match="needs url"):
        RedisTurnBackend(client=object())


# -- An entry written twice ---------------------------------------------------------


class _Posted:
    """What a process posted: (node, kind, body, ref), each sender's failure
    callback kept."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str, str, str]] = []
        self.refuse: set[str] = set()

    def __call__(
        self, node: str, kind: str, body: str, target: str, ref: str = "", *, failed: Any = None
    ) -> None:
        self.messages.append((node, kind, body, ref))
        if kind in self.refuse and failed is not None:
            self.refuse.discard(kind)
            failed(ConnectionError("OOM command not allowed"))

    def of(self, kind: str) -> list[tuple[str, str, str, str]]:
        return [m for m in self.messages if m[1] == kind]


class _Taker:
    """An executor of this process counting the turns and controls it got."""

    executor_id = "host-a"
    closed = False

    def __init__(self) -> None:
        self.taken: list[str] = []
        self.applied: list[str] = []

    def take(self, link: Any) -> None:
        self.taken.append(link.request.turn_id)

    async def control(self, message: Any) -> None:
        await asyncio.sleep(0)
        self.applied.append(message.type)


class _Owner:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.publications: list[str] = []

    async def room_call(self, message: RoomCall) -> HostResult:
        self.calls.append(message.verb)
        return HostResult(call_id=message.call_id)

    async def room_publication(self, message: RoomPublication) -> None:
        self.publications.append(message.turn_id)


def _routes() -> tuple[Routes, _Posted, list[asyncio.Task[None]]]:
    posted = _Posted()
    tasks: list[asyncio.Task[None]] = []
    routes = Routes(
        "node-b",
        posted,
        lambda work: tasks.append(asyncio.ensure_future(work)),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
    )
    return routes, posted, tasks


def test_a_request_written_twice_is_taken_once_and_answered_twice() -> None:
    routes, posted, _ = _routes()
    taker = _Taker()
    routes.shared["host-a"] = taker  # type: ignore[assignment]
    request = _request()
    asked = Envelope("request", "node-a", request.model_dump_json(), "host-a", "ref-1")

    routes.route(asked)
    routes.route(asked)

    assert taker.taken == [request.turn_id]
    assert posted.of("answer") == [("node-a", "answer", TAKEN, "ref-1")] * 2


async def test_a_channel_control_written_twice_is_applied_once_and_answered_twice() -> None:
    routes, posted, tasks = _routes()
    taker = _Taker()
    routes.shared["host-a"] = taker  # type: ignore[assignment]
    raw = ChannelControlEnvelope(message=RoomDetached(channel_id="agent", room_id="room"))
    asked = Envelope("channel", "node-a", raw.model_dump_json(), "host-a", "ref-2")

    routes.route(asked)
    routes.route(asked)  # while the first is applied: its answer is coming
    await asyncio.gather(*tasks)
    routes.route(asked)  # once answered: answered again

    assert taker.applied == ["room_detached"]
    assert [m[3] for m in posted.of("answer")] == ["ref-2", "ref-2"]


async def test_work_after_a_turn_written_twice_reaches_its_owner_once() -> None:
    routes, posted, tasks = _routes()
    owner = _Owner()
    routes.owner = owner  # type: ignore[assignment]
    call = RoomCall(
        channel_id="agent", room_id="room", executor_id="host-a", turn_id="t-1", verb="send"
    )
    published = RoomPublication(
        channel_id="agent",
        room_id="room",
        executor_id="host-a",
        turn_id="t-1",
        frame=Ephemeral(channel="room:room", event={}),
    )
    for message, ref in ((call, "ref-3"), (published, "ref-4")):
        body = OwnerMessageEnvelope(message=message).model_dump_json()
        routes.route(Envelope("owner", "node-a", body, "", ref))
        routes.route(Envelope("owner", "node-a", body, "", ref))
    await asyncio.gather(*tasks)

    assert owner.calls == ["send"]
    assert owner.publications == ["t-1"]


async def test_work_after_a_turn_redis_refused_is_written_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_redis_routes, "_WRITE_RETRY_DELAY", 0.01)
    routes, posted, _ = _routes()
    routes.shared["host-a"] = _Taker()  # type: ignore[assignment]
    request = _request()
    routes.route(Envelope("request", "node-a", request.model_dump_json(), "host-a", "ref-5"))
    posted.refuse.add("owner")

    await routes.port.room_publication(
        RoomPublication(
            channel_id="agent",
            room_id="room",
            executor_id="host-a",
            turn_id=request.turn_id,
            frame=Ephemeral(channel="room:room", event={}),
        )
    )
    await asyncio.sleep(0.1)

    written = posted.of("owner")
    assert len(written) == 2 and written[0][3] == written[1][3] != ""


# -- The view of the others ---------------------------------------------------------


class _Registry:
    def __init__(self) -> None:
        self.down = False

    async def view(self, forget_after_ms: int) -> View:
        if self.down:
            raise ConnectionError("Connection refused")
        return View(now=1_000, beats={"node-a:w": 1_000}, owners={"node-a": 1_000})

    async def running(self, turns: dict[str, list[str]]) -> dict[str, frozenset[str]]:
        return {}


async def test_the_others_unread_for_the_heartbeat_timeout_are_not_alive() -> None:
    """Redis unreachable: an owner stops waiting for an executor it cannot
    hear of, rather than trust the last view for good."""
    registry = _Registry()
    live = LiveExecutors(registry, 0.01, 0.1, dict)  # type: ignore[arg-type]
    live.start()
    try:
        await asyncio.sleep(0.05)
        assert live.alive("node-a:w") and live.owner_alive("node-a")

        registry.down = True
        await asyncio.sleep(0.3)

        assert not live.alive("node-a:w")
        assert not live.owner_alive("node-a")
    finally:
        await live.close()


# -- The writers --------------------------------------------------------------------

_TURN: contextvars.ContextVar[str] = contextvars.ContextVar("turn")


class _Client:
    """A Redis client whose pipelines take *delay* to write."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.written: list[str] = []
        self.contexts: list[str | None] = []

    def pipeline(self, transaction: bool) -> _Pipe:
        return _Pipe(self)


class _Pipe:
    def __init__(self, client: _Client) -> None:
        self._client = client
        self._bodies: list[str] = []

    def xadd(self, key: str, fields: dict[str, str]) -> None:
        self._bodies.append(fields["body"])

    def pexpire(self, key: str, ttl: int) -> None:
        return None

    async def execute(self) -> None:
        await asyncio.sleep(self._client.delay)
        self._client.contexts.append(_TURN.get(None))
        self._client.written.extend(self._bodies)


def _writers() -> list[asyncio.Task[Any]]:
    return [t for t in asyncio.all_tasks() if t.get_name().startswith("roomkit-turns-outbox-")]


def _message(body: str) -> Envelope:
    return Envelope("frame", "node-a", body, "turn-1")


async def test_a_writer_left_idle_goes_and_the_next_message_starts_another() -> None:
    client = _Client()
    outbox = Outbox(client, lambda node: f"inbox:{node}", ttl_ms=50)
    for node in ("a", "b", "c"):
        outbox.post(node, _message(node))
    await asyncio.sleep(0.2)

    assert _writers() == []

    outbox.post("a", _message("again"))
    await asyncio.sleep(0.02)

    assert client.written == ["a", "b", "c", "again"]
    await outbox.close(drain=1.0)


async def test_a_writer_carries_no_context_of_the_turn_that_posted_first() -> None:
    client = _Client()
    outbox = Outbox(client, lambda node: f"inbox:{node}", ttl_ms=60_000)
    _TURN.set("turn-1 span")
    outbox.post("a", _message("m"))
    await asyncio.sleep(0.02)

    assert client.contexts == [None]
    await outbox.close(drain=1.0)


async def test_closing_writes_what_is_queued() -> None:
    client = _Client(delay=0.02)
    outbox = Outbox(client, lambda node: f"inbox:{node}", ttl_ms=60_000)
    for body in ("m1", "m2", "m3"):
        outbox.post("a", _message(body))
        await asyncio.sleep(0)

    await outbox.close(drain=1.0)

    assert client.written == ["m1", "m2", "m3"]


async def test_closing_tells_the_senders_of_what_it_could_not_write() -> None:
    client = _Client(delay=1.0)
    outbox = Outbox(client, lambda node: f"inbox:{node}", ttl_ms=60_000)
    told: list[str] = []
    outbox.post("a", _message("m1"), lambda error: told.append("m1"))
    await asyncio.sleep(0.01)  # m1 is being written
    outbox.post("a", _message("m2"), lambda error: told.append("m2"))

    await outbox.close(drain=0.05)
    outbox.post("a", _message("m3"), lambda error: told.append("m3"))

    assert client.written == []
    assert told == ["m1", "m2", "m3"]
