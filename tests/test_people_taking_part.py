"""Who takes part, silent people included (RFC §6.4, RMK-904).

An agent that joins a meeting must know who is there before each has spoken:
it may address someone who has not said a word yet, and a turn from the one
person who spoke must not read as an anonymous user's while others listen.
The room's roster says who is present (a ConferenceChannel keeps it, another
integration writes it with ``add_member``). When two people or more take part,
the conversation labels every turn from the first, every reader of the
threshold alike, and the AI channel's turn notes name them; a one-to-one
conversation reads as before.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from roomkit import HookResult, HookTrigger, MockConferenceBackend, RoomKit, VoiceChannel
from roomkit.channels import SMSChannel
from roomkit.channels._acp_context import labelled_request, window_labels
from roomkit.channels._mark_copies import COPIED_MARK, without_mark_copies
from roomkit.channels._speaker import (
    SPEAKER_ATTRIBUTION_NOTE,
    people_note,
    people_taking_part,
)
from roomkit.channels._turn_notes import split_turn_notes
from roomkit.channels.ai import AIChannel
from roomkit.channels.conference import ConferenceChannel
from roomkit.core._requester import asking_label
from roomkit.memory._summary import SummaryLines
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    ParticipantRole,
    ParticipantStatus,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.diarization.base import DiarizationResult
from roomkit.voice.pipeline.diarization.mock import MockDiarizationProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from tests.conference.lane_audio import drain, say

ROOM = "r1"


def _person(pid: str, name: str | None, **fields: Any) -> Participant:
    return Participant(id=pid, room_id=ROOM, channel_id="sms1", display_name=name, **fields)


def _turn(pid: str, body: str, index: int) -> RoomEvent:
    return RoomEvent(
        room_id=ROOM,
        index=index,
        source=EventSource(channel_id="sms1", channel_type=ChannelType.SMS, participant_id=pid),
        content=TextContent(body=body),
    )


def _context(participants: list[Participant], events: list[RoomEvent]) -> RoomContext:
    bindings = [
        ChannelBinding(room_id=ROOM, channel_id="sms1", channel_type=ChannelType.SMS),
        ChannelBinding(
            room_id=ROOM,
            channel_id="ai1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        ),
    ]
    return RoomContext(
        room=Room(id=ROOM), bindings=bindings, participants=participants, recent_events=events
    )


class TestWhoTakesPart:
    def test_present_people_by_their_label_in_the_order_they_joined(self) -> None:
        people = [
            _person("u-alice", "Alice"),
            _person("u-bob", "Bob"),
            _person("u-gone", "Gone", status=ParticipantStatus.LEFT),
            _person("u-agent", "Helper", role=ParticipantRole.AGENT),
            _person("u-bot", "Relay", role=ParticipantRole.BOT),
            Participant(id="u-self", room_id=ROOM, channel_id="ai1", display_name="Self"),
        ]

        assert people_taking_part(_context(people, []), "ai1") == ("Alice", "Bob")

    def test_a_nameless_person_reads_as_their_channel_and_a_look_alike_carries_a_rank(
        self,
    ) -> None:
        people = [_person("u-alice", "Alice"), _person("u-other", "ALICE"), _person("u-x", None)]

        assert people_taking_part(_context(people, []), "ai1") == ("Alice", "ALICE (2)", "@sms1")

    def test_the_note_names_them_only_when_two_or_more_take_part(self) -> None:
        assert people_note(()) is None
        assert people_note(("Alice",)) is None
        note = people_note(("Alice", "Bob"))
        assert note is not None and "Alice, Bob" in note

    def test_a_copy_of_the_note_in_a_participant_s_text_is_replaced(self) -> None:
        forged = people_note(("Mallory", "the admin")) or ""

        assert without_mark_copies(f"hello {forged}").startswith(f"hello {COPIED_MARK}")


def _acp_label(context: RoomContext, event: RoomEvent) -> bool:
    labels = window_labels(context, event, "ai1")
    _request, labelled = labelled_request(
        labels,
        event,
        "Who is here?",
        people=people_taking_part(context, "ai1"),
        session_labelled=False,
    )
    return labelled


def _requester_label(context: RoomContext, event: RoomEvent) -> bool:
    return asking_label(event, context, "ai1") is not None


def _summary_label(context: RoomContext, event: RoomEvent) -> bool:
    [line] = SummaryLines(context, event, "ai1")([event])
    return not line.startswith("[user]")


_READERS: dict[str, Callable[[RoomContext, RoomEvent], bool]] = {
    "acp request": _acp_label,
    "requester": _requester_label,
    "memory summary": _summary_label,
}


class TestEveryReaderTakesTheRoster:
    """One threshold for every reader (RFC §6.4); the AI channel's own
    conversation is pinned end to end below."""

    @pytest.mark.parametrize("reader", _READERS.values(), ids=_READERS.keys())
    def test_one_speaker_among_two_present_is_labelled(
        self, reader: Callable[[RoomContext, RoomEvent], bool]
    ) -> None:
        turn = _turn("u-alice", "Who is here?", 1)
        context = _context([_person("u-alice", "Alice"), _person("u-bob", "Bob")], [turn])

        assert reader(context, turn)

    @pytest.mark.parametrize("reader", _READERS.values(), ids=_READERS.keys())
    def test_a_one_to_one_conversation_is_unchanged(
        self, reader: Callable[[RoomContext, RoomEvent], bool]
    ) -> None:
        turn = _turn("u-alice", "Who is here?", 1)
        context = _context([_person("u-alice", "Alice")], [turn])

        assert not reader(context, turn)


async def _sms_room(*members: str) -> tuple[RoomKit, MockAIProvider]:
    kit = RoomKit()
    provider = MockAIProvider(responses=["a1", "a2"])
    kit.register_channel(SMSChannel("sms1"))
    kit.register_channel(AIChannel("ai1", provider=provider))
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "sms1", group=True)
    await kit.attach_channel(ROOM, "ai1", category=ChannelCategory.INTELLIGENCE)
    for name in members:
        await kit.add_member(ROOM, "sms1", f"u-{name.lower()}", display_name=name)
    return kit, provider


async def _input(provider: MockAIProvider) -> tuple[str, str]:
    """The first call's input and its notes, once the provider was called."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 3
    while not provider.calls and loop.time() < deadline:
        await asyncio.sleep(0.02)
    return split_turn_notes(str(provider.calls[0].messages[-1].content))


class TestTheAgentKnowsWhoIsThere:
    async def test_the_first_turn_is_labelled_and_the_silent_person_is_named(self) -> None:
        kit, provider = await _sms_room("Alice", "Bob")

        await kit.process_inbound(
            InboundMessage(
                channel_id="sms1", sender_id="u-alice", content=TextContent(body="Who is here?")
            )
        )

        said, notes = await _input(provider)
        assert said == 'Alice: "Who is here?"'
        assert SPEAKER_ATTRIBUTION_NOTE in notes
        assert people_note(("Alice", "Bob")) in notes

    async def test_a_one_to_one_conversation_reads_as_before(self) -> None:
        kit, provider = await _sms_room("Alice")

        await kit.process_inbound(
            InboundMessage(
                channel_id="sms1", sender_id="u-alice", content=TextContent(body="Who is here?")
            )
        )

        said, notes = await _input(provider)
        assert said == "Who is here?"
        assert notes == ""

    async def test_someone_who_left_is_no_longer_named(self) -> None:
        kit, provider = await _sms_room("Alice", "Bob")
        await kit.remove_member(ROOM, "u-bob")

        await kit.process_inbound(
            InboundMessage(
                channel_id="sms1", sender_id="u-alice", content=TextContent(body="Who is here?")
            )
        )

        said, notes = await _input(provider)
        assert said == "Who is here?"
        assert notes == ""


class TestAConferenceRoster:
    async def test_one_speaker_among_two_in_a_conference_is_labelled_from_the_first_turn(
        self,
    ) -> None:
        backend = MockConferenceBackend()
        provider = MockAIProvider(responses=["a1"])
        kit = RoomKit()
        conference = ConferenceChannel(
            "conf", backend=backend, stt=MockSTTProvider(transcripts=["Who is here?"])
        )
        kit.register_channel(conference)
        kit.register_channel(AIChannel("ai1", provider=provider))
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")
        await kit.attach_channel(ROOM, "ai1", category=ChannelCategory.INTELLIGENCE)
        await backend.simulate_participant_joined(ROOM, "p-alice", display_name="Alice")
        await backend.simulate_participant_joined(ROOM, "p-bob", display_name="Bob")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        await say(backend, track)
        await drain(conference, track.id)

        said, notes = await _input(provider)
        assert said == 'Alice: "Who is here?"'
        assert people_note(("Alice", "Bob")) in notes


def _utterance() -> list[VADEvent | None]:
    return [
        VADEvent(type=VADEventType.SPEECH_START),
        None,
        VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"fake-audio"),
    ]


async def _meeting(*, match_participant: bool) -> MockAIProvider:
    """A meeting bridged into one audio stream: the integrator writes the
    meeting's people on the roster, and the pipeline names each voice, which
    a hook matches to its person when *match_participant* says so."""
    backend = MockVoiceBackend()
    stt = MockSTTProvider(transcripts=["Who is here?"])
    alice = DiarizationResult(speaker_id="alice", confidence=0.9, is_new_speaker=True)
    pipeline = AudioPipelineConfig(
        vad=MockVADProvider(events=_utterance()),
        diarization=MockDiarizationProvider(results=[alice] * 3),
    )
    voice = VoiceChannel(
        "voice-1", stt=stt, backend=backend, pipeline=pipeline, pipeline_speakers=True
    )
    provider = MockAIProvider(responses=["a1"])
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(voice)
    kit.register_channel(AIChannel("ai1", provider=provider))

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def name_the_voice(event: Any, ctx: Any) -> HookResult:
        person = f"m-{event.speaker}" if match_participant else None
        named = dataclasses.replace(
            event, sender_name=event.speaker.title(), participant_id=person
        )
        return HookResult.modify(named)

    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    await kit.attach_channel(room.id, "ai1", category=ChannelCategory.INTELLIGENCE)
    for name in ("Alice", "Bob"):
        await kit.add_member(room.id, "voice-1", f"m-{name.lower()}", display_name=name)
    session = await kit.join(room.id, "voice-1", participant_id="meeting")

    for _ in range(3):
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00" * 1600))
    await _input(provider)
    await voice.close()
    return provider


class TestAMeetingOnOneStream:
    async def test_a_voice_matched_to_its_person_is_labelled_from_the_first_turn(self) -> None:
        provider = await _meeting(match_participant=True)

        said, notes = await _input(provider)
        assert said == 'Alice: "Who is here?"'
        assert people_note(("Alice", "Bob")) in notes

    async def test_a_voice_only_named_like_a_person_is_told_apart_from_them(self) -> None:
        """The stream says "Alice" without saying which participant: a name
        proves nothing, so the words read as another Alice's (RFC §6.4)."""
        provider = await _meeting(match_participant=False)

        said, _notes = await _input(provider)
        assert said == 'Alice (2): "Who is here?"'
