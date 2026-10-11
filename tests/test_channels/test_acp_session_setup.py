"""What an ACP session is opened with (RFC §6.4, ACP agent channel, *Session setup*).

The room's MCP servers go in ``session/new``; ``session_config`` and the values
the host set on the room are set before the session is prompted, on every
session the channel opens for the room: the first, a rebuilt one, one opened
after the agent restarted, and a standalone turn's. A value the agent refuses
fails the opening: the agent's mode is what makes it ask RoomKit's gate before
acting, so a session left off it is ungated, not degraded.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import patch

import acp
import pytest
from acp.schema import (
    McpServerStdio,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SetSessionConfigOptionResponse,
)

from roomkit import ACPChannel, ACPSessionInvalidatedError
from roomkit.models.context import RoomContext
from roomkit.models.delivery import STANDALONE
from roomkit.models.enums import EventType
from roomkit.models.event import RoomEvent
from roomkit.models.room import Room
from tests.conftest import make_event
from tests.test_channels.test_acp import (
    _binding,
    _context,
    _FakeACPConnection,
    _FakeTransport,
    _prompt,
)

ROOM = "room-1"
OTHER_ROOM = "room-2"
DEFAULTS = {"model": "opus", "mode": "auto"}


def _select(config_id: str, current: str) -> SessionConfigOptionSelect:
    choices = ["opus", "sonnet"] if config_id == "model" else ["auto", "default", "plan"]
    return SessionConfigOptionSelect(
        id=config_id,
        name=config_id.title(),
        type="select",
        current_value=current,
        options=[SessionConfigSelectOption(value=value, name=value) for value in choices],
    )


class _Agent:
    """Keeps each session's config the way a real agent does.

    A new session starts on the agent's defaults (the shared fake keeps one
    config for every session, which would hide a session opened unconfigured).
    """

    def __init__(self, connection: _FakeACPConnection, *, refuses: str | None = None) -> None:
        # ``"mode"`` refuses any mode, ``"model=sonnet"`` that one value.
        self.connection = connection
        self.refuses = refuses
        self.sessions: dict[str, dict[str, str]] = {}
        self._open = connection.new_session
        connection.new_session = self.new_session  # type: ignore[method-assign]
        connection.set_config_option = self.set_config_option  # type: ignore[method-assign]

    def options(self, session_id: str) -> list[Any]:
        return [_select(key, value) for key, value in self.sessions[session_id].items()]

    async def new_session(self, **kwargs: Any) -> Any:
        response = await self._open(**kwargs)
        self.sessions[response.session_id] = dict(DEFAULTS)
        return response.model_copy(update={"config_options": self.options(response.session_id)})

    async def set_config_option(
        self, *, config_id: str, session_id: str, value: str | bool
    ) -> SetSessionConfigOptionResponse:
        self.connection.set_config_calls.append(
            {"config_id": config_id, "session_id": session_id, "value": value}
        )
        refused = self.refuses in (config_id, f"{config_id}={value}")
        if refused or config_id not in self.sessions[session_id]:
            raise acp.RequestError.invalid_params({"configId": config_id})
        self.sessions[session_id][config_id] = str(value)
        return SetSessionConfigOptionResponse(config_options=self.options(session_id))


def _setup(
    tmp_path: Any,
    *,
    mcp_servers: Any = None,
    session_config: dict[str, str | bool] | None = None,
    refuses: str | None = None,
) -> tuple[ACPChannel, _FakeACPConnection, _FakeTransport, _Agent]:
    connection = _FakeACPConnection(None, emit_updates=False)
    transport = _FakeTransport(connection)
    channel = ACPChannel(
        "acp-agent",
        transport=transport,
        cwd=tmp_path,
        mcp_servers=mcp_servers,
        session_config=session_config,
    )
    connection.client = channel._client
    return channel, connection, transport, _Agent(connection, refuses=refuses)


def _server(room_id: str) -> McpServerStdio:
    return McpServerStdio(
        name="notes",
        command="notes-server",
        args=[],
        env=[acp.schema.EnvVariable(name="NOTES_ROOM", value=room_id)],
    )


def _request(room_id: str = ROOM, *, standalone: bool = False) -> RoomEvent:
    event = make_event(room_id=room_id, body="request", index=0)
    if not standalone:
        return event
    return event.model_copy(update={"type": EventType.INSTRUCTION, "metadata": {STANDALONE: True}})


async def _turn(channel: ACPChannel, event: RoomEvent) -> None:
    context = RoomContext(
        room=Room(id=event.room_id), bindings=[_binding(event.room_id)], recent_events=[event]
    )
    output = await channel.on_event(event, _binding(event.room_id), context)
    assert output.response_stream is not None
    [chunk async for chunk in output.response_stream]


def _prompted_sessions(connection: _FakeACPConnection) -> list[str]:
    return [call["session_id"] for call in connection.prompt_calls]


class TestMCPServersPerRoom:
    async def test_a_list_goes_to_every_room(self, tmp_path: Any) -> None:
        shared = [_server("any")]
        channel, connection, _, _ = _setup(tmp_path, mcp_servers=shared)
        await _turn(channel, _request(ROOM))
        await _turn(channel, _request(OTHER_ROOM))

        assert [call["mcp_servers"] for call in connection.new_session_calls] == [shared, shared]
        await channel.close()

    @pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
    async def test_a_function_gives_each_room_its_servers(
        self, tmp_path: Any, asynchronous: bool
    ) -> None:
        asked: list[str] = []

        def servers(room_id: str) -> list[McpServerStdio]:
            asked.append(room_id)
            return [_server(room_id)]

        async def servers_async(room_id: str) -> list[McpServerStdio]:
            return servers(room_id)

        channel, connection, _, _ = _setup(
            tmp_path, mcp_servers=servers_async if asynchronous else servers
        )
        await _turn(channel, _request(ROOM))
        await _turn(channel, _request(OTHER_ROOM))

        assert asked == [ROOM, OTHER_ROOM]
        assert [call["mcp_servers"] for call in connection.new_session_calls] == [
            [_server(ROOM)],
            [_server(OTHER_ROOM)],
        ]
        await channel.close()

    async def test_the_function_is_asked_again_for_a_reopened_session(self, tmp_path: Any) -> None:
        asked: list[str] = []

        def servers(room_id: str) -> list[McpServerStdio]:
            asked.append(room_id)
            return [_server(room_id)]

        channel, connection, transport, _ = _setup(tmp_path, mcp_servers=servers)
        await _turn(channel, _request())
        transport.alive = False  # the agent restarted
        await _turn(channel, _request())

        assert asked == [ROOM, ROOM]
        assert connection.new_session_calls[1]["mcp_servers"] == [_server(ROOM)]
        await channel.close()

    async def test_any_iterable_of_servers_still_works(self, tmp_path: Any) -> None:
        servers = (_server(name) for name in ("a", "b"))  # a generator, not a function
        channel, connection, _, _ = _setup(tmp_path, mcp_servers=servers)
        await _turn(channel, _request(ROOM))
        await _turn(channel, _request(OTHER_ROOM))

        expected = [_server("a"), _server("b")]
        assert [call["mcp_servers"] for call in connection.new_session_calls] == [
            expected,
            expected,
        ]
        await channel.close()

    async def test_a_standalone_turn_gets_its_rooms_servers(self, tmp_path: Any) -> None:
        channel, connection, _, _ = _setup(tmp_path, mcp_servers=lambda room: [_server(room)])
        await _turn(channel, _request(ROOM, standalone=True))

        [call] = connection.new_session_calls
        assert call["mcp_servers"] == [_server(ROOM)]
        assert call["roomkit.live/sessionScope"] == "turn"
        await channel.close()


class TestSessionConfig:
    async def test_it_is_set_before_the_first_prompt(self, tmp_path: Any) -> None:
        channel, connection, _, agent = _setup(tmp_path, session_config={"mode": "default"})
        await _turn(channel, _request())

        assert connection.set_config_calls == [
            {"config_id": "mode", "session_id": "session-1", "value": "default"}
        ]
        assert agent.sessions["session-1"]["mode"] == "default"
        assert channel.session_config(ROOM) == {"model": "opus", "mode": "default"}
        await channel.close()

    async def test_a_value_the_session_already_has_is_not_set_again(self, tmp_path: Any) -> None:
        channel, connection, _, _ = _setup(tmp_path, session_config={"model": "opus"})
        await _turn(channel, _request())

        assert connection.set_config_calls == []
        await channel.close()

    async def test_it_and_the_hosts_values_survive_an_agent_restart(self, tmp_path: Any) -> None:
        channel, _connection, transport, agent = _setup(
            tmp_path, session_config={"mode": "default"}
        )
        await _turn(channel, _request())
        await channel.set_config_option(ROOM, "model", "sonnet")

        transport.alive = False  # the agent died; its sessions with it
        await _turn(channel, _request())

        assert channel.session_id(ROOM) == "session-2"
        assert agent.sessions["session-2"] == {"model": "sonnet", "mode": "default"}
        assert channel.session_config(ROOM) == {"model": "sonnet", "mode": "default"}
        await channel.close()

    async def test_it_and_the_hosts_values_survive_a_rebuilt_session(self, tmp_path: Any) -> None:
        channel, connection, _, agent = _setup(tmp_path, session_config={"mode": "default"})
        await channel.set_config_option(ROOM, "model", "sonnet")
        prompt = connection.prompt
        refused: list[str] = []

        async def refuses_once(session_id: str, *args: Any, **kwargs: Any) -> Any:
            if not refused:
                refused.append(session_id)
                raise ACPSessionInvalidatedError("unusable", recovery_authorized=True)
            return await prompt(session_id, *args, **kwargs)

        with patch.object(connection, "prompt", refuses_once):
            await _turn(channel, _request())

        assert refused == ["session-1"]
        assert _prompted_sessions(connection) == ["session-2"]
        assert agent.sessions["session-2"] == {"model": "sonnet", "mode": "default"}
        await channel.close()

    async def test_a_standalone_turn_takes_it_without_a_room_session(self, tmp_path: Any) -> None:
        channel, connection, _, agent = _setup(tmp_path, session_config={"mode": "default"})
        await _turn(channel, _request(standalone=True))

        assert channel.session_id(ROOM) is None
        assert _prompted_sessions(connection) == ["session-1"]
        assert agent.sessions["session-1"]["mode"] == "default"
        await channel.close()

    async def test_a_standalone_turn_after_a_restart_takes_the_hosts_values(
        self, tmp_path: Any
    ) -> None:
        channel, connection, transport, agent = _setup(
            tmp_path, session_config={"mode": "default"}
        )
        await channel.set_config_option(ROOM, "model", "sonnet")
        transport.alive = False
        await _turn(channel, _request(standalone=True))

        assert _prompted_sessions(connection) == ["session-2"]
        assert agent.sessions["session-2"] == {"model": "sonnet", "mode": "default"}
        await channel.close()

    async def test_setting_another_value_replaces_one_the_agent_refuses(
        self, tmp_path: Any
    ) -> None:
        channel, connection, transport, agent = _setup(tmp_path)
        await channel.set_config_option(ROOM, "model", "sonnet")
        transport.alive = False
        agent.refuses = "model=sonnet"  # the restarted agent no longer has that model
        with pytest.raises(RuntimeError, match="set_config_option"):
            await _turn(channel, _request())

        assert await channel.set_config_option(ROOM, "model", "opus") == DEFAULTS
        await _turn(channel, _request())

        assert _prompted_sessions(connection) == ["session-3"]
        assert agent.sessions["session-3"]["model"] == "opus"
        await channel.close()

    async def test_a_value_set_while_the_session_closes_is_not_kept(self, tmp_path: Any) -> None:
        channel, _connection, _, agent = _setup(tmp_path)
        await _turn(channel, _request())
        answering, answer = asyncio.Event(), asyncio.Event()
        set_option = agent.set_config_option

        async def slow(**kwargs: Any) -> Any:
            answering.set()
            await answer.wait()
            return await set_option(**kwargs)

        with patch.object(agent.connection, "set_config_option", slow):
            setting = asyncio.create_task(channel.set_config_option(ROOM, "model", "sonnet"))
            await answering.wait()
            await channel.close_session(ROOM)
            answer.set()
            await setting

        assert channel._room_config == {}
        assert channel._session_options == {}
        await channel.close()

    async def test_close_session_forgets_what_the_host_set(self, tmp_path: Any) -> None:
        channel, _connection, _, agent = _setup(tmp_path, session_config={"mode": "default"})
        await channel.set_config_option(ROOM, "model", "sonnet")

        await channel.close_session(ROOM)
        await _turn(channel, _request())

        assert agent.sessions["session-2"] == {"model": "opus", "mode": "default"}
        await channel.close()

    async def test_the_hosts_values_are_per_room(self, tmp_path: Any) -> None:
        channel, _connection, _, agent = _setup(tmp_path)
        await channel.set_config_option(ROOM, "model", "sonnet")
        await _turn(channel, _request(OTHER_ROOM))

        assert agent.sessions["session-2"] == DEFAULTS
        await channel.close()

    def test_values_acp_cannot_carry_are_refused(self, tmp_path: Any) -> None:
        with pytest.raises(TypeError, match="session_config"):
            ACPChannel("acp", ["agent"], cwd=tmp_path, session_config={"effort": 3})  # type: ignore[dict-item]


class TestARefusedValueFailsClosed:
    """The session is released and never prompted; the error names the option."""

    @pytest.mark.parametrize("standalone", [False, True], ids=["room", "standalone"])
    async def test_a_turn_fails_and_prompts_nothing(self, tmp_path: Any, standalone: bool) -> None:
        channel, connection, _, _ = _setup(
            tmp_path, session_config={"mode": "default"}, refuses="mode"
        )
        with pytest.raises(RuntimeError, match="mode='default'"):
            await _prompt(channel, _request(standalone=standalone), _context(_request()))

        assert connection.prompt_calls == []
        assert connection.closed_sessions == ["session-1"]
        assert channel._sessions == channel._turn_sessions == channel._session_rooms == {}
        assert all(not lock.locked() for lock in channel._room_locks.values())
        await channel.close()

    async def test_set_config_option_fails_when_it_opens_the_session(self, tmp_path: Any) -> None:
        channel, connection, _, _ = _setup(
            tmp_path, session_config={"mode": "default"}, refuses="mode"
        )
        with pytest.raises(RuntimeError, match="mode='default'"):
            await channel.set_config_option(ROOM, "model", "sonnet")

        assert [call["config_id"] for call in connection.set_config_calls] == ["mode"]
        assert channel.session_id(ROOM) is None
        assert connection.closed_sessions == ["session-1"]
        await channel.close()

    async def test_a_rebuilt_session_that_refuses_fails_the_turn(self, tmp_path: Any) -> None:
        channel, connection, _, agent = _setup(tmp_path, session_config={"mode": "default"})
        prompt = connection.prompt

        async def refuses_once(session_id: str, *args: Any, **kwargs: Any) -> Any:
            if session_id == "session-1":
                agent.refuses = "mode"  # the session the recovery opens refuses it
                raise ACPSessionInvalidatedError("unusable", recovery_authorized=True)
            return await prompt(session_id, *args, **kwargs)

        with (
            patch.object(connection, "prompt", refuses_once),
            pytest.raises(RuntimeError, match="mode='default'"),
        ):
            await _turn(channel, _request())

        assert connection.closed_sessions == ["session-1", "session-2"]
        assert channel._sessions == channel._session_rooms == {}
        assert all(not lock.locked() for lock in channel._room_locks.values())
        await channel.close()

    async def test_a_lost_connection_is_not_reported_as_a_refusal(self, tmp_path: Any) -> None:
        channel, connection, _, _ = _setup(tmp_path, session_config={"mode": "default"})

        async def gone(**kwargs: Any) -> Any:
            raise ConnectionError("agent gone")

        with patch.object(connection, "set_config_option", gone), pytest.raises(ConnectionError):
            await _turn(channel, _request())

        assert connection.prompt_calls == []
        await channel.close()

    async def test_a_session_the_agent_cannot_close_is_logged(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        channel, connection, _, _ = _setup(
            tmp_path, session_config={"mode": "default"}, refuses="mode"
        )
        connection.takes_session_close = False
        with (
            caplog.at_level(logging.WARNING, logger="roomkit.channels.acp"),
            pytest.raises(RuntimeError),
        ):
            await _turn(channel, _request())

        assert connection.closed_sessions == []
        assert "session-1 was not set up and stays open" in caplog.text
        await channel.close()

    async def test_a_refused_host_value_fails_the_rooms_next_session(self, tmp_path: Any) -> None:
        channel, connection, transport, agent = _setup(tmp_path)
        await channel.set_config_option(ROOM, "model", "sonnet")

        transport.alive = False
        agent.refuses = "model"  # the restarted agent no longer has that model
        with pytest.raises(RuntimeError, match="model='sonnet'"):
            await _prompt(channel, _request(), _context(_request()))

        assert connection.prompt_calls == []
        await channel.close()


class TestOneOpeningAtATime:
    """A room's session is opened once, whoever asks for it first (RFC §6.4, item 2)."""

    @staticmethod
    def _gated(connection: _FakeACPConnection) -> asyncio.Event:
        """Hold every room-scope ``session/new`` until the returned event is set."""
        gate = asyncio.Event()
        opener = connection.new_session

        async def held(**kwargs: Any) -> Any:
            if kwargs.get("roomkit.live/sessionScope") == "room":
                await gate.wait()
            return await opener(**kwargs)

        connection.new_session = held  # type: ignore[method-assign]
        return gate

    @staticmethod
    def _room_opens(connection: _FakeACPConnection) -> int:
        return sum(
            call.get("roomkit.live/sessionScope") == "room"
            for call in connection.new_session_calls
        )

    async def test_two_host_calls_during_a_standalone_turn_open_one_session(
        self, tmp_path: Any
    ) -> None:
        channel, connection, _, agent = _setup(tmp_path)
        gate = self._gated(connection)
        prompting, finish = asyncio.Event(), asyncio.Event()
        prompt = connection.prompt

        async def holds(session_id: str, *args: Any, **kwargs: Any) -> Any:
            prompting.set()
            await finish.wait()
            return await prompt(session_id, *args, **kwargs)

        with patch.object(connection, "prompt", holds):
            turn = asyncio.create_task(_turn(channel, _request(standalone=True)))
            await prompting.wait()  # the standalone turn holds the room's turn lock
            model = asyncio.create_task(channel.set_config_option(ROOM, "model", "sonnet"))
            mode = asyncio.create_task(channel.set_config_option(ROOM, "mode", "default"))
            await asyncio.sleep(0.01)
            gate.set()
            await asyncio.gather(model, mode)
            finish.set()
            await turn

        assert self._room_opens(connection) == 1
        room_session = channel.session_id(ROOM)
        assert agent.sessions[room_session] == {"model": "sonnet", "mode": "default"}
        assert set(channel._session_rooms) == {room_session}  # no orphan session
        await channel.close()

    async def test_a_host_call_and_a_first_prompt_open_one_session(self, tmp_path: Any) -> None:
        channel, connection, _, agent = _setup(tmp_path)
        gate = self._gated(connection)

        setting = asyncio.create_task(channel.set_config_option(ROOM, "model", "sonnet"))
        turn = asyncio.create_task(_turn(channel, _request()))
        await asyncio.sleep(0.01)
        gate.set()
        await asyncio.gather(setting, turn)

        assert self._room_opens(connection) == 1
        assert _prompted_sessions(connection) == ["session-1"]
        assert agent.sessions["session-1"]["model"] == "sonnet"
        await channel.close()

    async def test_close_session_waits_for_an_opening_then_closes_it(self, tmp_path: Any) -> None:
        channel, connection, _, _ = _setup(tmp_path)
        gate = self._gated(connection)

        setting = asyncio.create_task(channel.set_config_option(ROOM, "model", "sonnet"))
        await asyncio.sleep(0.01)
        closing = asyncio.create_task(channel.close_session(ROOM))
        await asyncio.sleep(0.01)
        assert not closing.done()  # it waits for the opening rather than miss it
        gate.set()
        await setting

        assert await closing is True
        assert connection.closed_sessions == ["session-1"]
        assert channel.session_id(ROOM) is None
        assert channel._room_locks == channel._opening_locks == {}
        await channel.close()


class TestACancelReachesATurnWhileItsSessionOpens:
    """A turn holds the room from the start, its session's opening included (item 7)."""

    @staticmethod
    async def _cancelled_while(
        channel: ACPChannel, event: RoomEvent, reached: asyncio.Event
    ) -> Any:
        """Start a turn, cancel the room once *reached* is set, and drain the turn."""
        output = await channel.on_event(event, _binding(event.room_id), _context(event))
        assert output.response_stream is not None
        drain = asyncio.create_task(_drain(output.response_stream))
        await reached.wait()
        assert channel._prompting(event.room_id)
        assert channel.active_turns == 1
        assert await channel.cancel(event.room_id) is True
        await drain
        return output

    @staticmethod
    def _hang(target: Any, name: str, reached: asyncio.Event) -> None:
        async def hangs(*args: Any, **kwargs: Any) -> Any:
            reached.set()
            await asyncio.sleep(3600)

        setattr(target, name, hangs)

    def _assert_never_prompted(self, channel: ACPChannel, connection: Any, output: Any) -> None:
        assert connection.prompt_calls == []
        assert output.response_metadata["acp"]["interrupted"] is True
        assert channel._turns == {} and channel.active_turns == 0
        assert all(not lock.locked() for lock in channel._room_locks.values())

    async def test_while_the_rooms_session_is_created(self, tmp_path: Any) -> None:
        channel, connection, _, _ = _setup(tmp_path)
        reached = asyncio.Event()
        self._hang(connection, "new_session", reached)

        output = await self._cancelled_while(channel, _request(), reached)

        self._assert_never_prompted(channel, connection, output)
        assert channel.session_id(ROOM) is None
        await channel.close()

    @pytest.mark.parametrize("standalone", [False, True], ids=["room", "standalone"])
    async def test_while_the_session_is_set_up(self, tmp_path: Any, standalone: bool) -> None:
        channel, connection, _, _ = _setup(tmp_path, session_config={"mode": "default"})
        reached = asyncio.Event()
        self._hang(connection, "set_config_option", reached)

        output = await self._cancelled_while(channel, _request(standalone=standalone), reached)

        self._assert_never_prompted(channel, connection, output)
        assert connection.closed_sessions == ["session-1"]
        assert connection.cancelled_sessions == []  # nothing was prompting
        assert channel._sessions == channel._turn_sessions == channel._session_rooms == {}
        await channel.close()

    async def test_while_the_agent_is_started(self, tmp_path: Any) -> None:
        channel, connection, transport, _ = _setup(tmp_path)
        reached = asyncio.Event()
        self._hang(transport, "open", reached)

        output = await self._cancelled_while(channel, _request(), reached)

        self._assert_never_prompted(channel, connection, output)
        assert channel._connection is None
        await channel.close()

    async def test_a_cancel_landing_as_the_opening_ends_closes_the_turn_session(
        self, tmp_path: Any
    ) -> None:
        channel, connection, _, agent = _setup(tmp_path, session_config={"mode": "default"})
        set_option = agent.set_config_option

        async def cancels_as_it_answers(**kwargs: Any) -> Any:
            # Scheduled now, it runs once the opening has finished and before
            # the turn wakes up to prompt.
            asyncio.ensure_future(channel.cancel(ROOM))
            return await set_option(**kwargs)

        connection.set_config_option = cancels_as_it_answers  # type: ignore[method-assign]
        output = await channel.on_event(_request(standalone=True), _binding(), _context())
        assert output.response_stream is not None
        await _drain(output.response_stream)

        self._assert_never_prompted(channel, connection, output)
        assert connection.closed_sessions == ["session-1"]
        assert channel._turn_sessions == {}
        await channel.close()


async def _drain(stream: Any) -> None:
    [chunk async for chunk in stream]
