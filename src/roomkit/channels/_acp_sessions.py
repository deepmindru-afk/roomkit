"""Which ACP session a turn talks to, and how that session lives.

A room gets one session, opened lazily on its first prompt and kept for the
room's life: the agent holds the room's conversation inside its own process.
A standalone turn (RFC §10.1.1 step 7) cannot empty that session, so it gets
one of its own, opened for the turn and never prompted again after it: closed
where the agent announces ``session/close``, kept by the agent until the
connection closes where it does not.
Both name their room to ``session/new`` and say which of the two they are: a
relay that files sessions by room would otherwise take the turn's for the
room's, and close it. They are serialized on the room's turn lock, and told
apart here by :meth:`ACPSessionsMixin._is_room_session`.

Both are set up the same way before they are prompted (RFC §6.4, ACP agent
channel, *Session setup*): the room's MCP servers in ``session/new``, then the
channel's ``session_config`` and the values the host set on the room. A value
the agent refuses fails the opening rather than leave a session off its setup.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from roomkit.channels._acp_client import _config_values, _model_dump

logger = logging.getLogger("roomkit.channels.acp")

MCPServersForRoom = Callable[[str], Sequence[Any] | Awaitable[Sequence[Any]]]
"""The MCP servers a room's sessions get, from the room's id (sync or async)."""


def _servers_source(
    mcp_servers: Sequence[Any] | MCPServersForRoom | None,
) -> list[Any] | MCPServersForRoom:
    """``mcp_servers`` as the channel keeps it: the function, or a copy of the list."""
    if callable(mcp_servers):
        return cast(MCPServersForRoom, mcp_servers)
    return list(mcp_servers or ())


def _session_setup(session_config: Mapping[str, str | bool] | None) -> dict[str, str | bool]:
    """``session_config`` checked and copied: ACP config values are strings or booleans."""
    setup = dict(session_config or {})
    for config_id, value in setup.items():
        if not isinstance(config_id, str) or not config_id or not isinstance(value, (str, bool)):
            raise TypeError(
                f"session_config maps an ACP config id to a string or a boolean, "
                f"got {config_id!r}: {value!r}"
            )
    return setup


def _reported_options(response: Any) -> list[Any]:
    """The config options an ACP response reports, as a list (empty when none)."""
    options = _model_dump(getattr(response, "config_options", None))
    return options if isinstance(options, list) else []


class ACPSessionsMixin:
    """Open, configure, serialize and close room and turn sessions."""

    channel_id: str
    _cwd: str
    _additional_directories: list[Path | str]
    _mcp_servers: list[Any] | MCPServersForRoom
    _session_setup: dict[str, str | bool]
    _room_config: dict[str, dict[str, str | bool]]
    _room_locks: dict[str, asyncio.Lock]
    _sessions: dict[str, str]
    _turn_sessions: dict[str, str]
    _session_rooms: dict[str, str]
    _session_options: dict[str, list[Any]]
    _prompted_index: dict[str, int]
    _labelled_rooms: set[str]
    _instructed_rooms: set[str]
    _agent_closes_sessions: bool

    # Implemented elsewhere on the channel. Declared as annotations, never as
    # stub methods: a stub here would shadow the implementation of any mixin
    # after this one in the MRO. A ``Coroutine``, not an ``Awaitable``, for the
    # reason ``ACPTurnMixin`` gives.
    session_config: Callable[[str], dict[str, str | bool]]
    _publish_config_options: Callable[[str, Any, dict[str, str | bool]], Coroutine[Any, Any, None]]

    async def _new_session(
        self, room_id: str, connection: Any, scope: Literal["room", "turn"]
    ) -> Any:
        """``session/new`` as this channel declares every session, room or turn."""
        return await connection.new_session(
            cwd=self._cwd,
            additional_directories=self._additional_directories or None,
            mcp_servers=await self._servers_for(room_id),
            **{"roomkit.live/roomId": room_id, "roomkit.live/sessionScope": scope},
        )

    async def _servers_for(self, room_id: str) -> list[Any]:
        """The MCP servers a session of *room_id* is given."""
        servers = self._mcp_servers
        if isinstance(servers, list):
            return list(servers)
        computed = servers(room_id)
        resolved: Sequence[Any] = await computed if inspect.isawaitable(computed) else computed
        return list(resolved or ())

    async def _configure_session(
        self,
        connection: Any,
        session_id: str,
        room_id: str,
        options: list[Any],
        overlay: Mapping[str, str | bool] | None = None,
    ) -> list[Any]:
        """Set on a new session each value its room's setup wants and it lacks.

        The setup is ``session_config`` overlaid with what the host set on the
        room, then with *overlay*, the value a host call opening the session is
        setting (it replaces the room's earlier one). Returns the options the
        agent last reported. A value the agent refuses raises: the session is
        left for the caller to release.
        """
        wanted = {**self._session_setup, **self._room_config.get(room_id, {}), **(overlay or {})}
        for config_id, value in wanted.items():
            if _config_values(options).get(config_id) == value:
                continue
            try:
                response = await connection.set_config_option(
                    config_id=config_id, session_id=session_id, value=value
                )
            except ConnectionError:
                raise  # the agent is gone, it refused nothing
            except Exception as exc:
                raise RuntimeError(
                    f"ACP session for room {room_id!r} could not take {config_id}={value!r} "
                    f"({self.channel_id}); it is not prompted. Set another value with "
                    "set_config_option, or close_session to forget the room's"
                ) from exc
            options = _reported_options(response) or options
        return options

    async def _session_for(
        self,
        room_id: str,
        connection: Any,
        overlay: Mapping[str, str | bool] | None = None,
    ) -> str:
        """The room's session, opened and set up if it has none (*overlay* as set up)."""
        session_id = self._sessions.get(room_id)
        if session_id is not None:
            return session_id
        response = await self._new_session(room_id, connection, "room")
        session_id = response.session_id
        try:
            options = await self._configure_session(
                connection, session_id, room_id, _reported_options(response), overlay
            )
        except BaseException:
            await self._release_unset_session(connection, session_id)
            raise
        self._sessions[room_id] = session_id
        self._session_rooms[session_id] = room_id
        self._session_options[session_id] = options
        await self._publish_config_options(
            session_id,
            self._session_options[session_id],
            _config_values(self._session_options[session_id]),
        )
        return session_id

    async def _open_for_host(
        self, room_id: str, connection: Any, overlay: Mapping[str, str | bool]
    ) -> tuple[str, bool]:
        """The room's session for a host call, set up with *overlay* if this call opens it.

        Returns the session and whether this call opened it: a first prompt
        that got the lock first opened it without *overlay*.
        """
        if room_id in self._turn_sessions:
            # A standalone turn is in flight and holds the room's lock, so this
            # caller may be running inside it (a tool handler): waiting would
            # deadlock. That turn never opens the room's session, so nothing
            # races this one. The setting is the room's, not the turn's.
            opened = room_id not in self._sessions
            return await self._session_for(room_id, connection, overlay), opened
        # Session creation is serialized on the room's turn lock so a
        # concurrent first prompt cannot open a second session.
        async with self._room_turn_lock(room_id):
            opened = room_id not in self._sessions
            return await self._session_for(room_id, connection, overlay), opened

    async def _discard_room_session(self, room_id: str, connection: Any) -> bool:
        """Forget a room session while its caller holds the turn lock.

        Keep the lock registered: a recovery must exclude other prompts until
        its replacement turn finishes. Only the public close retires the lock.
        """
        session_id = self._sessions.pop(room_id, None)
        self._prompted_index.pop(room_id, None)
        self._labelled_rooms.discard(room_id)
        self._instructed_rooms.discard(room_id)
        if session_id is None:
            return False
        self._session_rooms.pop(session_id, None)
        self._session_options.pop(session_id, None)
        if connection is not None:
            await self._release_session(connection, session_id)
        return True

    def _options_for(self, room_id: str) -> list[Any]:
        session_id = self._sessions.get(room_id)
        if session_id is None:
            return []
        return self._session_options.get(session_id, [])

    @contextlib.asynccontextmanager
    async def _room_turn_lock(self, room_id: str) -> AsyncGenerator[None, None]:
        """Hold the room's turn lock, tolerating its retirement.

        ``close_session`` drops the entry while still holding the lock, so
        the map does not keep one lock per room the channel ever served. A
        coroutine that was queued on the retired lock therefore wakes owning
        an object nobody else can reach: it releases and retries on the
        current one. Without that re-check a fresh caller would take a
        brand-new lock and run the critical section alongside the waiter —
        which is how a room ends up with two sessions.
        """
        while True:
            lock = self._room_locks.setdefault(room_id, asyncio.Lock())
            await lock.acquire()
            if self._room_locks.get(room_id) is lock:
                break
            lock.release()
        try:
            yield
        finally:
            lock.release()

    def _is_room_session(self, session_id: str) -> bool:
        """Whether *session_id* is a room's session — not a turn's, not a closed one.

        What a session reports about itself (usage, tunables) describes the
        room only when it is the room's: a turn session lives for one turn, and
        announcing its state would read as the room's changing.
        """
        return session_id in self._sessions.values()

    async def _open_turn_session(self, room_id: str, connection: Any) -> str:
        """Open the session a standalone turn runs in.

        It takes the room's setup as the room's next session would, a refused
        value failing the turn, then the room session's current configuration
        (a model the host chose, a mode) where the agent accepts it: the turn
        is blank on history, not on how the channel was set up. A session
        this opens is closed here if anything interrupts the setup,
        cancellation included, so no half-open session outlives it.
        """
        response = await self._new_session(room_id, connection, "turn")
        session_id = response.session_id
        if session_id in self._session_rooms:
            # A relay that ignores the scope answers with the room's session:
            # prompting it would tell the room, and closing it after the turn
            # would end the room's conversation. The turn fails, the room lives.
            raise RuntimeError(
                f"ACP session/new for a standalone turn in room {room_id!r} returned "
                f"open session {session_id!r}; the transport must file a 'turn' "
                "session apart from the room's"
            )
        self._turn_sessions[room_id] = session_id
        self._session_rooms[session_id] = room_id
        try:
            options = await self._configure_session(
                connection, session_id, room_id, _reported_options(response)
            )
            fresh = _config_values(options)
            for config_id, value in self.session_config(room_id).items():
                if fresh.get(config_id) != value:
                    await self._copy_config(connection, session_id, config_id, value)
        except BaseException:
            await self._close_turn_session(room_id, session_id, connection)
            raise
        return session_id

    async def _copy_config(
        self, connection: Any, session_id: str, config_id: str, value: str | bool
    ) -> None:
        try:
            await connection.set_config_option(
                config_id=config_id, session_id=session_id, value=value
            )
        except Exception:
            # "Where the agent accepts it": a value this agent refuses leaves
            # the turn on its default, which is logged rather than fatal.
            logger.warning(
                "ACP standalone turn could not take %r=%r from the room's session (%s)",
                config_id,
                value,
                self.channel_id,
                exc_info=True,
            )

    async def _close_turn_session(self, room_id: str, session_id: str, connection: Any) -> None:
        """Forget a standalone turn's session and close it where the agent can. Never raises."""
        if self._turn_sessions.get(room_id) == session_id:
            self._turn_sessions.pop(room_id, None)
        self._session_rooms.pop(session_id, None)
        if not self._agent_closes_sessions:
            logger.warning(
                "ACP standalone turn session %s stays open in the agent until the connection "
                "closes: the agent does not announce session/close (%s)",
                session_id,
                self.channel_id,
            )
        await self._release_session(connection, session_id)

    async def _release_unset_session(self, connection: Any, session_id: str) -> None:
        """Release a room session whose setup failed. Never raises."""
        if not self._agent_closes_sessions:
            logger.warning(
                "ACP session %s was not set up and stays open in the agent until the "
                "connection closes: the agent does not announce session/close (%s)",
                session_id,
                self.channel_id,
            )
        await self._release_session(connection, session_id)

    async def _release_session(self, connection: Any, session_id: str) -> None:
        """Ask the agent to close *session_id*, if it takes ``session/close``. Never raises.

        The channel has already forgotten the session: an agent that refuses,
        or cannot be asked, keeps it until the connection closes, and no
        caller is better served by an exception than by that.
        """
        if not self._agent_closes_sessions:
            return
        try:
            await connection.close_session(session_id)
        except ConnectionError:
            # The connection is going away, and the session with it.
            logger.debug(
                "ACP session %s close: connection closed (%s)", session_id, self.channel_id
            )
        except Exception:
            # The agent announced session/close and still refused this one.
            logger.warning(
                "ACP session %s close failed; it stays open in the agent (%s)",
                session_id,
                self.channel_id,
                exc_info=True,
            )
