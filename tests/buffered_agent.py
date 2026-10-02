"""An agent that answers with buffered events, for the framework's buffered path.

An ``AIChannel`` answers every turn through its tool loop's stream (RFC §6.4).
The framework's buffered response path stays for every other channel that
answers at once (an orchestration strategy's result, a host's own agent), and
a test of that path drives it with this one.
"""

from __future__ import annotations

from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent


class BufferedAgent(Channel):
    """Answers each event with the next *rows* of its *answers*, as buffered
    messages, cycling through them.

    ``asked`` records every event it was asked to answer.
    """

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, channel_id: str, *answers: str, rows: int = 1) -> None:
        super().__init__(channel_id)
        self._answers = list(answers) or ["ok"]
        self._rows = rows
        self._next = 0
        self.asked: list[RoomEvent] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError("an agent takes no inbound message")

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()
        self.asked.append(event)
        answers = [self._answer(event) for _ in range(self._rows)]
        return ChannelOutput(responded=True, response_events=answers)

    def _answer(self, event: RoomEvent) -> RoomEvent:
        body = self._answers[self._next % len(self._answers)]
        self._next += 1
        return RoomEvent(
            room_id=event.room_id,
            source=EventSource(channel_id=self.channel_id, channel_type=ChannelType.AI),
            content=TextContent(body=body),
            chain_depth=event.chain_depth + 1,
        )
