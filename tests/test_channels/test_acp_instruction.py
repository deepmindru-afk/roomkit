"""An INSTRUCTION reaching an ACP agent (RFC §10.1.1).

An ACP session holds its history inside the agent's process, so the rules an
in-process channel keeps by rebuilding its context each turn are kept here by
what the prompt says and by which session receives it.
"""

from __future__ import annotations

from typing import Any

from roomkit.models.enums import EventType
from roomkit.models.event import RoomEvent
from tests.conftest import make_event
from tests.test_channels.test_acp import _channel, _context, _prompt, _sent

ROOM = "room-1"


def _instruction(body: str) -> RoomEvent:
    # Never committed, so it carries no index of its own (index 0).
    return make_event(room_id=ROOM, body=body, index=0).model_copy(
        update={"type": EventType.INSTRUCTION}
    )


async def test_an_instruction_moves_the_cursor_past_what_it_caught_up_on(tmp_path: Any) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    alpha = make_event(room_id=ROOM, body="alpha said", index=0)
    beta = make_event(room_id=ROOM, body="beta said", index=1)
    await _prompt(channel, _instruction("Summarize the meeting."), _context(alpha, beta))

    following = make_event(room_id=ROOM, body="next request", index=2)
    await _prompt(channel, following, _context(alpha, beta, following))

    assert "beta said" in _sent(connection, turn=0)
    assert _sent(connection, turn=1) == "next request"
    await channel.close()
