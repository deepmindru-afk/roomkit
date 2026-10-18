"""The Redis turn backend's parts that need no Redis (RFC §10.6.7, §10.6.8):
its inbox entries, what an executor declares, how an owner counts a beat an
executor makes outside the turn's messages, and the optional dependency."""

from __future__ import annotations

import asyncio
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
from roomkit.turns import InMemoryTurnBackend, TurnExecutor, TurnTimeouts
from roomkit.turns._owner_turn import OwnerTurn
from roomkit.turns._redis_inbox import Envelope
from roomkit.turns._redis_presence import Declaration
from roomkit.turns.owner import registry_turn_client, turn_request
from roomkit.turns.wire import Accepted
from tests.private_module import load_private_copy

QUICK = TurnTimeouts(heartbeat_interval=0.05, heartbeat_timeout=0.2, accept=0.2, cancel=0.2)


def test_an_inbox_entry_reads_back_from_bytes_or_text() -> None:
    sent = Envelope("frame", "node-a", '{"x": 1}', target="turn-1", ref="")
    as_bytes = {key.encode(): value.encode() for key, value in sent.fields().items()}

    assert Envelope.parse(as_bytes) == sent
    assert Envelope.parse(sent.fields()) == sent


def test_a_declaration_takes_its_channels_with_every_room_hook_it_defines() -> None:
    declared = Declaration("node-a", frozenset({"meteo"}), frozenset({"audit"}))

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
