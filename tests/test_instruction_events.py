"""An INSTRUCTION event directs an agent and is never a participant's words (RFC §10.1.1).

The property under test: whatever path an instruction takes through the inbound
pipeline — delivered, refused, blocked by a hook — the room's timeline never
holds it, no transport receives it, and the model reads it as the application's
direction for one turn, not as something someone in the room said.
"""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from roomkit import AIChannel, HookResult, HookTrigger, RoomKit
from roomkit.channels.base import Channel
from roomkit.memory import BudgetAwareMemory, MemoryProvider, MemoryResult, SlidingWindowMemory
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.mock import MockAIProvider

ROOM = "r-instruction"
INSTRUCTION = "Handoff complete. Introduce yourself to the caller."
FINGERPRINT = {"sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(), "length": 51}


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


async def _kit(
    *, streaming: bool = False, memory: MemoryProvider | None = None
) -> tuple[RoomKit, Speaker, MockAIProvider]:
    """A room where the instruction enters on ``voice`` and ``speaker`` is another transport.

    The source channel never receives its own event, so what a transport is
    handed is read on a second one.
    """
    kit = RoomKit()
    speaker = Speaker("speaker")
    provider = MockAIProvider(["Bonjour, je suis Paul."], streaming=streaming)
    kit.register_channel(Speaker("voice"))
    kit.register_channel(speaker)
    kit.register_channel(AIChannel("agent", provider=provider, memory=memory))
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
    # A fingerprint, never the text: the reply is stored and fanned out.
    assert stored[0].metadata["instruction"] == FINGERPRINT
    assert all(INSTRUCTION not in e.model_dump_json() for e in stored)
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
        ("agent", FINGERPRINT)
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


class _CountingMemory(BudgetAwareMemory):
    """A budget with a floor that keeps events whatever the view: an empty view
    is not a blank page, only not calling it is."""

    def __init__(self) -> None:
        super().__init__(SlidingWindowMemory(), max_context_tokens=100_000, min_events=3)
        self.retrieved = 0

    async def retrieve(self, *args: object, **kwargs: object) -> MemoryResult:
        self.retrieved += 1
        return await super().retrieve(*args, **kwargs)  # type: ignore[arg-type]


async def _talking_room(
    *, streaming: bool = False
) -> tuple[RoomKit, MockAIProvider, _CountingMemory]:
    """A room where the caller and the agent already exchanged a line, and the
    agent used a tool whose result its working memory quotes."""
    memory = _CountingMemory()
    kit, _speaker, provider = await _kit(streaming=streaming, memory=memory)
    agent = kit.get_channel("agent")
    agent._tool_usage.record(ROOM, "lookup_customer", {"id": 7}, "PREVIOUS-TOOL-RESULT")  # type: ignore[union-attr]
    await kit.process_inbound(
        InboundMessage(channel_id="voice", sender_id="caller", content=TextContent(body="Allô")),
        room_id=ROOM,
    )
    assert memory.retrieved == 1
    return kit, provider, memory


async def _send_instruction(kit: RoomKit, via: str, **fields: object) -> None:
    if via == "process_inbound":
        await kit.process_inbound(_instruction(**fields), room_id=ROOM)
        return
    await kit.send_event(
        ROOM,
        "voice",
        TextContent(body=INSTRUCTION),
        event_type=EventType.INSTRUCTION,
        addressed_to=["agent"],
        **fields,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("via", ["process_inbound", "send_event"])
async def test_a_standalone_instruction_reads_nothing_of_the_room(via: str, streaming: bool):
    """RFC §10.1.1 step 7: no history, no memory provider call, and none of the
    room's working memories (here the tool-usage digest) in the system prompt."""
    kit, provider, memory = await _talking_room(streaming=streaming)

    await _send_instruction(kit, via, standalone=True)

    assert memory.retrieved == 1
    [only] = provider.calls[-1].messages
    assert only.role == "user" and INSTRUCTION in str(only.content)
    assert "Allô" not in str(only.content)
    assert "PREVIOUS-TOOL-RESULT" not in (provider.calls[-1].system_prompt or "")
    stored = await _messages(kit)
    assert stored[-1].metadata["instruction"] == FINGERPRINT
    assert all("standalone" not in e.metadata for e in stored)
    await kit.close()


@pytest.mark.parametrize("via", ["process_inbound", "send_event"])
async def test_an_instruction_without_standalone_reads_the_room(via: str):
    """A metadata key of the same name is not the flag: only the typed field is."""
    kit, provider, memory = await _talking_room()

    await _send_instruction(kit, via, metadata={"standalone": True})

    assert memory.retrieved == 2
    contents = [str(m.content) for m in provider.calls[-1].messages]
    assert any("Allô" in c for c in contents) and INSTRUCTION in contents[-1]
    assert "PREVIOUS-TOOL-RESULT" in (provider.calls[-1].system_prompt or "")
    await kit.close()


async def test_standalone_is_refused_on_anything_but_an_instruction():
    """Set on a message, it would land as a participant's line without the isolation asked."""
    kit, speaker, provider = await _kit()

    with pytest.raises(ValidationError):
        InboundMessage(
            channel_id="voice", sender_id="caller", content=TextContent(body="x"), standalone=True
        )
    with pytest.raises(ValueError):
        await kit.send_event(ROOM, "voice", TextContent(body="x"), standalone=True)

    assert await _messages(kit) == []
    assert provider.calls == [] and speaker.delivered == []
    await kit.close()
