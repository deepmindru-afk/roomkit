"""The session a standalone ACP turn runs in (RFC §10.1.1 step 7).

An ACP session holds the room's conversation inside the agent's process and
cannot be emptied, so a turn that must start from a blank page gets a session
of its own: opened for the turn, closed after it, never the room's. What lives
here is that lifecycle, and the one fact the rest of the channel asks of it —
whether a session is the room's.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from roomkit.channels._acp_client import _config_values

logger = logging.getLogger("roomkit.channels.acp")


class ACPTurnSessionMixin:
    """Open, configure and close standalone turn sessions."""

    channel_id: str
    _cwd: str
    _additional_directories: list[Path | str]
    _mcp_servers: list[Any]
    _sessions: dict[str, str]
    _turn_sessions: dict[str, str]
    _session_rooms: dict[str, str]

    def session_config(self, room_id: str) -> dict[str, str | bool]:
        raise NotImplementedError

    async def _new_session(self, room_id: str, connection: Any) -> Any:
        """``session/new`` as this channel declares every session, room or turn."""
        return await connection.new_session(
            cwd=self._cwd,
            additional_directories=self._additional_directories or None,
            mcp_servers=self._mcp_servers,
            **{"roomkit.live/roomId": room_id},
        )

    def _is_room_session(self, session_id: str) -> bool:
        """Whether *session_id* is a room's session — not a turn's, not a closed one.

        What a session reports about itself (usage, tunables) describes the
        room only when it is the room's: a turn session lives for one turn, and
        announcing its state would read as the room's changing.
        """
        return session_id in self._sessions.values()

    async def _open_turn_session(self, room_id: str, connection: Any) -> str:
        """Open the session a standalone turn runs in.

        It takes the room session's current configuration (a model the host
        chose, a mode) where the agent accepts it: the turn is blank on
        history, not on how the channel was set up. A session this opens is
        closed here if anything interrupts the setup, cancellation included,
        so no half-open session outlives it.
        """
        response = await self._new_session(room_id, connection)
        session_id = response.session_id
        self._turn_sessions[room_id] = session_id
        self._session_rooms[session_id] = room_id
        try:
            fresh = _config_values(getattr(response, "config_options", None))
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
        """Close and forget a standalone turn's session. Never raises: the turn is over."""
        if self._turn_sessions.get(room_id) == session_id:
            self._turn_sessions.pop(room_id, None)
        self._session_rooms.pop(session_id, None)
        try:
            await connection.close_session(session_id)
        except Exception:
            logger.debug(
                "ACP standalone session close failed (%s)", self.channel_id, exc_info=True
            )
