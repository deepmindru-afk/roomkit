"""An ACP agent across the turn backend (RFC §10.6.2, §10.6.3, §10.6.8, RMK-812).

The host's session verbs are channel calls reaching the executor that holds
the room's session, and the values it sets belong to the room; a lost
executor's room reopens its session on the next one, caught up and set up
again; a detach closes the room's session; a steering ``Cancel`` cancels the
prompt as the channel's own cancel does.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import ACPChannel, RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, HookExecution, HookTrigger
from roomkit.models.event import TextContent
from roomkit.models.steering import Cancel
from roomkit.turns import InMemoryTurnBackend, TurnExecutor
from roomkit.turns._rooms import room_state_key
from tests.test_channels.test_acp import _channel, _FakeACPConnection, _FakeTransport, _sent
from tests.test_channels.test_acp_discussion import _Hanging
from tests.test_framework import SimpleChannel

ROOM = "room-1"
AGENT = "acp-agent"


async def _room(channel: ACPChannel, backend: InMemoryTurnBackend | None = None) -> RoomKit:
    kit = RoomKit(turn_backend=backend)
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "sms", category=ChannelCategory.TRANSPORT)
    await kit.attach_channel(ROOM, AGENT, category=ChannelCategory.INTELLIGENCE)
    return kit


async def _say(kit: RoomKit, body: str) -> None:
    await kit.process_inbound(
        InboundMessage(
            channel_id="sms",
            sender_id="marie",
            content=TextContent(body=body),
            addressed_to=[AGENT],
        ),
        room_id=ROOM,
    )
    await asyncio.sleep(0)


def _hanging(tmp_path: Any) -> tuple[ACPChannel, _Hanging]:
    connection = _Hanging()
    channel = ACPChannel(AGENT, transport=_FakeTransport(connection), cwd=tmp_path)
    connection.client = channel._client
    return channel, connection


# -- The host's session verbs --------------------------------------------------------


async def test_a_host_s_session_values_belong_to_the_room(tmp_path: Any) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    kit = await _room(channel)

    values = await kit.channel_call(
        AGENT, ROOM, "set_config_option", config_id="model", value="sonnet"
    )

    assert values == {"model": "sonnet"}
    assert await kit.channel_call(AGENT, ROOM, "session_config") == {"model": "sonnet"}
    (option,) = await kit.channel_call(AGENT, ROOM, "config_options")
    assert option["id"] == "model" and "sonnet" in str(option)
    room = await kit.get_room(ROOM)
    assert room.metadata[room_state_key(AGENT)] == {"config": {"model": "sonnet"}}

    assert await kit.channel_call(AGENT, ROOM, "close_session") is True
    room = await kit.get_room(ROOM)
    assert room_state_key(AGENT) not in room.metadata
    assert connection.closed_sessions == ["session-1"]
    await kit.close()


async def test_a_verb_an_acp_agent_lacks_is_refused(tmp_path: Any) -> None:
    channel, _connection, _ = _channel(tmp_path, emit_updates=False)
    kit = await _room(channel)
    with pytest.raises(ValueError, match="set_config_option"):
        await kit.channel_call(AGENT, ROOM, "prompt", text="hi")
    await kit.close()


# -- A lost executor ---------------------------------------------------------------


async def test_a_lost_executor_s_room_reopens_its_session_on_the_next(tmp_path: Any) -> None:
    backend = InMemoryTurnBackend()
    remote, remote_connection, _ = _channel(tmp_path, emit_updates=False)
    elsewhere = TurnExecutor(
        lambda channel_id: remote if channel_id == AGENT else None,
        hosts=lambda channel_id: channel_id == AGENT,
        executor_id="elsewhere",
    )
    # Served before the kit's own executor: the room's session opens there.
    await backend.serve(elsewhere)
    local, local_connection, _ = _channel(tmp_path, emit_updates=False)
    kit = await _room(local, backend)

    await kit.channel_call(AGENT, ROOM, "set_config_option", config_id="model", value="sonnet")
    await _say(kit, "first question")
    assert len(remote_connection.prompt_calls) == 1
    assert not local_connection.prompt_calls

    await elsewhere.close()
    await _say(kit, "second question")

    # The kit's executor opened the room's session anew: set to the value the
    # room keeps, and caught up on what the lost session had been told.
    assert len(local_connection.new_session_calls) == 1
    assert [(c["config_id"], c["value"]) for c in local_connection.set_config_calls] == [
        ("model", "sonnet")
    ]
    sent = _sent(local_connection)
    assert "first question" in sent and sent.endswith("second question")
    await kit.close()
    await remote.close()


# -- Detaching -------------------------------------------------------------------------


async def test_detaching_an_acp_agent_closes_the_room_s_session(tmp_path: Any) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    kit = await _room(channel)
    await _say(kit, "hello")
    session = channel.session_id(ROOM)
    assert session is not None

    await kit.detach_channel(ROOM, AGENT)

    assert connection.closed_sessions == [session]
    assert channel.session_id(ROOM) is None
    await kit.attach_channel(ROOM, AGENT, category=ChannelCategory.INTELLIGENCE)
    await _say(kit, "hello again")
    assert len(connection.new_session_calls) == 2
    await kit.close()


async def test_a_detach_never_waits_for_the_room_s_running_turn(tmp_path: Any) -> None:
    channel, connection = _hanging(tmp_path)
    kit = await _room(channel)
    said = asyncio.create_task(_say(kit, "read every file"))
    await asyncio.wait_for(connection.prompted.wait(), 5)
    session = channel.session_id(ROOM)

    await asyncio.wait_for(kit.detach_channel(ROOM, AGENT), 2)
    assert connection.closed_sessions == []

    connection.release.set()
    await asyncio.wait_for(said, 5)
    for _ in range(100):
        if connection.closed_sessions:
            break
        await asyncio.sleep(0.01)
    assert connection.closed_sessions == [session]
    await kit.close()


# -- Steering ------------------------------------------------------------------------


async def test_a_steering_cancel_cancels_the_prompt_as_its_own_cancel_does(
    tmp_path: Any,
) -> None:
    channel, connection = _hanging(tmp_path)
    kit = await _room(channel)
    responses: list[Any] = []

    @kit.turn_executor.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def observe(event: Any, ctx: Any) -> None:
        responses.append(event)

    said = asyncio.create_task(_say(kit, "read every file"))
    await asyncio.wait_for(connection.prompted.wait(), 5)

    reached = await kit.steer(Cancel(reason="hold"), AGENT, room_id=ROOM)
    await asyncio.wait_for(said, 5)
    await asyncio.sleep(0.05)

    assert reached == 1
    assert connection.cancelled_sessions == [channel.session_id(ROOM)]
    # The prompt returned: the turn ended as the agent ended it, not cut.
    assert len(responses) == 1
    await kit.close()


async def test_a_steering_cancel_reaches_no_idle_acp_agent(tmp_path: Any) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    kit = await _room(channel)
    await _say(kit, "hello")
    assert await kit.steer(Cancel(reason="hold"), AGENT, room_id=ROOM) == 0
    assert connection.cancelled_sessions == []
    await kit.close()


class _SlowOpening(_FakeACPConnection):
    """An agent whose ``session/new`` waits until released."""

    def __init__(self) -> None:
        super().__init__(None, emit_updates=False)
        self.opening = asyncio.Event()
        self.release = asyncio.Event()

    async def new_session(self, **kwargs: Any) -> Any:
        self.opening.set()
        await self.release.wait()
        return await super().new_session(**kwargs)


async def test_a_steering_cancel_while_the_session_opens_stops_the_prompt(
    tmp_path: Any,
) -> None:
    connection = _SlowOpening()
    channel = ACPChannel(AGENT, transport=_FakeTransport(connection), cwd=tmp_path)
    connection.client = channel._client
    kit = await _room(channel)
    said = asyncio.create_task(_say(kit, "hello"))
    await asyncio.wait_for(connection.opening.wait(), 5)

    assert await kit.steer(Cancel(reason="hold"), AGENT, room_id=ROOM) == 1
    connection.release.set()
    await asyncio.wait_for(said, 5)

    # Nothing was prompted: the opening was stopped before a prompt could go.
    assert connection.prompt_calls == []
    assert not channel._prompting(ROOM)
    await kit.close()
