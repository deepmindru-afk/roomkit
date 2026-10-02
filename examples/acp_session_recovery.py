"""Recover a refused ACP prompt once, using a local demonstration transport.

Run: uv run python examples/acp_session_recovery.py
Requires: roomkit[acp]. No server, credentials or external agent is contacted.

The fake host reserves one retry in memory before refusing the first prompt.
A real host must durably reserve it for the event before authorizing recovery.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import acp
from acp.schema import InitializeResponse, NewSessionResponse, PromptResponse

from roomkit import ACPChannel, ACPSessionInvalidatedError, ACPTransport
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room

logger = logging.getLogger("roomkit.examples.acp_recovery")


class DemoTransport(ACPTransport):
    """Both ends of a canned connection; no wire protocol is invented here."""

    def __init__(self) -> None:
        self.client: Any = None
        self.sessions = 0
        self.reserved = False
        self.events: list[str] = []

    @property
    def name(self) -> str:
        return "recovery-demo"

    async def open(self, client: Any, *, queue: Any) -> Any:
        self.client = client
        return self

    async def initialize(self, protocol_version: int, **_kwargs: Any) -> InitializeResponse:
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(self, **_kwargs: Any) -> NewSessionResponse:
        self.sessions += 1
        return NewSessionResponse(session_id=f"demo-{self.sessions}")

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        self.events.append(kwargs["roomkit.live/eventId"])
        if not self.reserved:
            self.reserved = True  # Demonstration only: production hosts persist this.
            raise ACPSessionInvalidatedError("session lost", recovery_authorized=True)
        await self.client.session_update(
            session_id, acp.update_agent_message_text("Served once in the new session.")
        )
        return PromptResponse(stop_reason="end_turn")

    async def close(self) -> None:
        pass


async def main() -> None:
    transport = DemoTransport()
    channel = ACPChannel("agent", transport=transport, cwd=Path.cwd())
    event = RoomEvent(
        room_id="demo",
        source=EventSource(channel_id="human", channel_type=ChannelType.CLI),
        content=TextContent(body="Continue this conversation."),
    )
    try:
        output = await channel.on_event(
            event,
            ChannelBinding(channel_id="agent", room_id="demo", channel_type=ChannelType.AI),
            RoomContext(room=Room(id="demo"), recent_events=[event]),
        )
        answer = "".join(
            [chunk async for chunk in output.response_stream if isinstance(chunk, str)]
        )
        assert transport.sessions == 2
        assert transport.events == [event.id, event.id]
        assert "interrupted" not in output.response_metadata["acp"]
        logger.info("%s Sessions: %d; one event, two attempts.", answer, transport.sessions)
    finally:
        await channel.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
