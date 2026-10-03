"""What an AI channel's transcript says of an event whose content extracts to nothing.

Without a describer such an event is omitted, from the history and from the
turn's input alike. ``AIChannel(describe_empty_event=...)`` gives the host the
word: it is asked only when the extraction is empty, its text stands in for
the event in the transcript, and ``None`` keeps the omission. The stored event
is never touched.
"""

from __future__ import annotations

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import MediaContent, RoomEvent
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIContext, AIMessage
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import respond

_UPLOAD = "The user sent attachments without a caption."


class _Describer:
    """Describes an upload stored with an empty body, and records what it read."""

    def __init__(self) -> None:
        self.read: list[str] = []

    def __call__(self, event: RoomEvent) -> str | None:
        self.read.append(event.id)
        return _UPLOAD if event.metadata.get("attachments") else None


def _upload() -> RoomEvent:
    return make_event(body="", channel_id="sms1", metadata={"attachments": [{"name": "a.pdf"}]})


async def _turn(
    event: RoomEvent, history: list[RoomEvent], **kwargs: object
) -> tuple[AIContext, AIChannel]:
    provider = MockAIProvider(responses=["ok"])
    ch = AIChannel("ai1", provider=provider, **kwargs)  # type: ignore[arg-type]
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    # The member's channel is bound too: an unbound source's events are
    # visible to no other channel.
    member = ChannelBinding(channel_id="sms1", room_id="r1", channel_type=ChannelType.SMS)
    context = RoomContext(
        room=Room(id="r1"), bindings=[member, binding], recent_events=[*history, event]
    )
    await respond(ch, event, binding, context)
    return provider.calls[0], ch


def _texts(messages: list[AIMessage]) -> list[str]:
    return [m.content for m in messages if isinstance(m.content, str)]


async def test_an_upload_without_a_caption_is_described_in_the_history_and_the_input() -> None:
    describer = _Describer()
    upload = _upload()

    sent, _ch = await _turn(upload, [_upload()], describe_empty_event=describer)

    assert sum(_UPLOAD in text for text in _texts(sent.messages)) == 2
    assert sent.messages[-1].role == "user"
    assert _UPLOAD in str(sent.messages[-1].content)
    # The stored event keeps its empty body: only the transcript reads the text.
    assert upload.content.body == ""


async def test_the_describer_is_asked_only_of_an_event_that_extracts_to_nothing() -> None:
    describer = _Describer()
    said = make_event(body="hello", channel_id="sms1")

    await _turn(
        said, [make_event(body="earlier", channel_id="sms1")], describe_empty_event=describer
    )

    assert describer.read == []


async def test_none_keeps_the_omission() -> None:
    describer = _Describer()
    empty = make_event(body="", channel_id="sms1")

    sent, _ch = await _turn(
        make_event(body="hello", channel_id="sms1"), [empty], describe_empty_event=describer
    )

    assert describer.read == [empty.id]
    assert "" not in _texts(sent.messages)
    assert [m.content for m in sent.messages if m.role == "user"] == ["hello"]


async def test_an_image_a_text_only_provider_cannot_read_is_described_too() -> None:
    """The extraction is empty for a captionless image on a provider without
    vision, whatever the reason: the describer covers it."""
    image = make_event(channel_id="sms1").model_copy(
        update={"content": MediaContent(url="https://x.test/a.png", mime_type="image/png")}
    )

    sent, _ch = await _turn(
        make_event(body="what was that?", channel_id="sms1"),
        [image],
        describe_empty_event=lambda event: "[an image]",
    )

    assert "[an image]" in _texts(sent.messages)


async def test_without_a_describer_an_empty_event_is_omitted() -> None:
    sent, _ch = await _turn(make_event(body="hello", channel_id="sms1"), [_upload()])

    assert [m.content for m in sent.messages if m.role == "user"] == ["hello"]
