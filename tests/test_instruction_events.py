"""An INSTRUCTION event directs an agent and is never a participant's words (RFC §10.1.1).

The property under test: whatever path an instruction takes through the inbound
pipeline — delivered, refused, blocked by a hook — the room's timeline never
holds it, no transport receives it, and the model reads it as the application's
direction for one turn, not as something someone in the room said.
"""

from __future__ import annotations

import pytest

from roomkit import AIChannel, HookResult, HookTrigger, RoomKit
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.mock import MockAIProvider

ROOM = "r-instruction"
INSTRUCTION = "Handoff complete. Introduce yourself to the caller."


class Speaker(Channel):
    """A transport that records what it was asked to deliver."""

    channel_type = ChannelType.VOICE

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.delivered: list[str] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.delivered.append(getattr(event.content, "body", ""))
        return ChannelOutput.empty()


async def _kit(*, streaming: bool = False) -> tuple[RoomKit, Speaker, MockAIProvider]:
    """A room where the instruction enters on ``voice`` and ``speaker`` is another transport.

    The source channel never receives its own event, so what a transport is
    handed is read on a second one.
    """
    kit = RoomKit()
    speaker = Speaker("speaker")
    provider = MockAIProvider(["Bonjour, je suis Paul."], streaming=streaming)
    kit.register_channel(Speaker("voice"))
    kit.register_channel(speaker)
    kit.register_channel(AIChannel("agent", provider=provider))
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "voice", category=ChannelCategory.TRANSPORT)
    await kit.attach_channel(ROOM, "speaker", category=ChannelCategory.TRANSPORT)
    await kit.attach_channel(ROOM, "agent", category=ChannelCategory.INTELLIGENCE)
    return kit, speaker, provider


def _instruction(**overrides: object) -> InboundMessage:
    fields: dict[str, object] = {
        "channel_id": "voice",
        "sender_id": "system",
        "event_type": EventType.INSTRUCTION,
        "content": TextContent(body=INSTRUCTION),
        "addressed_to": ["agent"],
    }
    fields.update(overrides)
    return InboundMessage(**fields)  # type: ignore[arg-type]


async def _messages(kit: RoomKit) -> list[RoomEvent]:
    """Every stored event but the bindings', BLOCKED audit records included."""
    events = await kit.store.list_events(ROOM, event_filter=EventFilter(include_blocked=True))
    return [e for e in events if e.type != EventType.CHANNEL_ATTACHED]


def _last_input(provider: MockAIProvider) -> str:
    content = provider.calls[-1].messages[-1].content
    return content if isinstance(content, str) else str(content)


@pytest.mark.parametrize("streaming", [False, True])
async def test_the_room_holds_the_agents_reply_and_never_the_instruction(streaming: bool):
    kit, speaker, provider = await _kit(streaming=streaming)

    result = await kit.process_inbound(_instruction(), room_id=ROOM)

    assert not result.blocked
    stored = await _messages(kit)
    assert [(e.type, e.source.channel_id, e.content.body) for e in stored] == [
        (EventType.MESSAGE, "agent", "Bonjour, je suis Paul.")
    ]
    assert stored[0].metadata["instruction"] == INSTRUCTION
    assert speaker.delivered == ["Bonjour, je suis Paul."]
    room = await kit.get_room(ROOM)
    assert room.event_count == len(await kit.store.list_events(ROOM))
    await kit.close()


async def test_the_model_reads_it_as_the_applications_direction_for_one_turn():
    kit, _speaker, provider = await _kit()

    await kit.process_inbound(_instruction(), room_id=ROOM)
    directed = _last_input(provider)
    assert INSTRUCTION in directed
    assert directed != INSTRUCTION  # marked, never passed off as a participant's line

    await kit.process_inbound(
        InboundMessage(channel_id="voice", sender_id="caller", content=TextContent(body="Merci")),
        room_id=ROOM,
    )
    later = provider.calls[-1].messages
    assert all(INSTRUCTION not in str(m.content) for m in later)
    await kit.close()


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"addressed_to": None}, "instruction_unaddressed"),
        ({"addressed_to": []}, "instruction_unaddressed"),
        ({"idempotency_key": "k1"}, "instruction_not_idempotent"),
    ],
)
async def test_a_refused_instruction_leaves_nothing_behind(overrides, reason):
    kit, speaker, provider = await _kit()

    result = await kit.process_inbound(_instruction(**overrides), room_id=ROOM)

    assert result.blocked and result.reason == reason
    assert await _messages(kit) == []
    assert provider.calls == [] and speaker.delivered == []
    await kit.close()


async def test_a_hook_blocked_instruction_is_not_stored():
    kit, speaker, provider = await _kit()

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def refuse(event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult.block("not now")

    result = await kit.process_inbound(_instruction(), room_id=ROOM)

    assert result.blocked
    assert await _messages(kit) == []
    assert provider.calls == [] and speaker.delivered == []
    await kit.close()


async def test_no_transport_receives_it_whatever_visibility_the_caller_asked():
    kit, speaker, _provider = await _kit()

    await kit.process_inbound(_instruction(visibility="all"), room_id=ROOM)

    assert INSTRUCTION not in speaker.delivered
    await kit.close()


async def test_send_event_directs_an_agent_the_same_way():
    kit, speaker, _provider = await _kit()

    await kit.send_event(
        ROOM,
        "voice",
        TextContent(body=INSTRUCTION),
        event_type=EventType.INSTRUCTION,
        addressed_to=["agent"],
    )

    stored = await _messages(kit)
    assert [(e.source.channel_id, e.metadata.get("instruction")) for e in stored] == [
        ("agent", INSTRUCTION)
    ]
    assert speaker.delivered == ["Bonjour, je suis Paul."]
    await kit.close()


@pytest.mark.parametrize(
    "overrides", [{"addressed_to": None}, {"addressed_to": ["agent"], "idempotency_key": "k"}]
)
async def test_send_event_raises_for_an_instruction_it_would_refuse(overrides):
    """Its contract is the committed event: a refusal has nothing to return."""
    kit, speaker, provider = await _kit()

    with pytest.raises(ValueError):
        await kit.send_event(
            ROOM,
            "voice",
            TextContent(body=INSTRUCTION),
            event_type=EventType.INSTRUCTION,
            **overrides,
        )

    assert await _messages(kit) == []
    assert provider.calls == [] and speaker.delivered == []
    await kit.close()
