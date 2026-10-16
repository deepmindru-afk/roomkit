"""The media an application keeps beside a message's content (RFC §6.4).

An application may store a message as its words, with its files in a record of
its own (the event's metadata, say). It hands the framework an
``attached_media`` reader naming them as media, and every AI channel reads the
event with them as parts after its own content: an image among them follows
the rules of an author's image (the resolver, the bound, the refusals, a
mention for a model that does not see), while the store and every broadcast
keep the event as it was stored.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence

import pytest

from roomkit import RoomKit
from roomkit.channels._turn_notes import TURN_IMAGES_HEADER
from roomkit.channels.ai import AIChannel
from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.memory.token_estimator import estimate_event_tokens
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.models.event import CompositeContent, MediaContent, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIContext, AIImagePart, AITextPart, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.utils import to_data_uri
from tests.test_author_images import (
    _BINDING,
    _MEMBER,
    _OK,
    _PNG,
    _event,
    _Host,
    _images,
    _MediaWatcher,
    _Scripted,
    _text,
)
from tests.test_framework import SimpleChannel
from tests.tool_loop_modes import read_reply, respond

_REF = "app-media://photo-1"


def _files(*names: str) -> dict[str, object]:
    """An application's own record of a message's files, beside its words."""
    return {"files": [{"ref": f"app-media://{name}", "name": f"{name}.png"} for name in names]}


def _reader(event: RoomEvent) -> list[MediaContent]:
    """The application's reading of that record: each file as a media reference."""
    return [
        MediaContent(url=f["ref"], mime_type="image/png", filename=f["name"])
        for f in event.metadata.get("files", [])
    ]


def _message(body: str, *files: str) -> RoomEvent:
    event = _event(TextContent(body=body))
    return event.model_copy(update={"metadata": _files(*files)})


class _KeepsMetadata(SimpleChannel):
    """A transport that stores what the sender's application said of a message."""

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        event = await super().handle_inbound(message, context)
        return event.model_copy(update={"metadata": {**event.metadata, **message.metadata}})


def _channel(
    streaming: bool = False,
    *,
    vision: bool = True,
    host: _Host | None = None,
    reader: object = _reader,
    provider: MockAIProvider | None = None,
    **kwargs: object,
) -> AIChannel:
    provider = provider or MockAIProvider(responses=["ok"] * 5, streaming=streaming, vision=vision)
    channel = AIChannel("ai1", provider=provider, **kwargs)  # type: ignore[arg-type]
    # The framework wires both at registration.
    channel._media_resolver = host if host is not None else _Host()  # noqa: SLF001
    channel._attached_media = reader  # type: ignore[assignment]  # noqa: SLF001
    return channel


async def _turn(
    channel: AIChannel, event: RoomEvent, history: Sequence[RoomEvent] = ()
) -> AIContext:
    context = RoomContext(
        room=Room(id="r1"), bindings=[_MEMBER, _BINDING], recent_events=[*history, event]
    )
    await respond(channel, event, _BINDING, context)
    return channel.provider.calls[-1]  # type: ignore[attr-defined]


class TestAttached:
    async def test_an_attached_image_reaches_the_model_with_its_words(
        self, streaming: bool
    ) -> None:
        host = _Host()

        context = await _turn(
            _channel(streaming, host=host), _message("compare with mine", "photo-1")
        )

        content = context.messages[-1].content
        assert isinstance(content, list)
        assert isinstance(content[0], AITextPart)
        assert content[0].text.startswith("compare with mine")
        assert isinstance(content[1], AIImagePart) and content[1].url == _PNG
        assert host.asked == [(_REF, "r1")]

    async def test_the_store_and_broadcasts_keep_the_event_as_stored(self) -> None:
        host = _Host()
        provider = MockAIProvider(responses=["a nice lamp"], vision=True)
        kit = RoomKit(media_resolver=host, attached_media=_reader)
        watcher = _MediaWatcher("watcher")
        kit.register_channel(_KeepsMetadata("user"))
        kit.register_channel(watcher)
        kit.register_channel(AIChannel("ai1", provider=provider))
        try:
            await kit.create_room(room_id="room")
            await kit.attach_channel("room", "user")
            await kit.attach_channel("room", "watcher")
            await kit.attach_channel("room", "ai1", category=ChannelCategory.INTELLIGENCE)
            message = InboundMessage(
                channel_id="user",
                sender_id="person",
                content=TextContent(body="compare with mine"),
                metadata=_files("photo-1"),
            )

            await kit.process_inbound(message, room_id="room")

            assert _images(provider.calls[-1]) == [_PNG]
            stored = [e for e in await kit.store.list_events("room") if e.metadata.get("files")]
            assert [e.content for e in stored] == [TextContent(body="compare with mine")]
            delivered = [e.content for e in watcher.delivered if e.metadata.get("files")]
            assert delivered == [TextContent(body="compare with mine")]
        finally:
            await kit.close()

    async def test_the_image_is_still_there_on_the_next_turn(self, streaming: bool) -> None:
        host = _Host()
        photo = _message("here is my lamp", "photo-1")

        context = await _turn(
            _channel(streaming, host=host),
            _event(TextContent(body="what colour is its foot?")),
            [photo],
        )

        assert _images(context) == [_PNG]
        assert context.messages[-1].content == "what colour is its foot?"

    async def test_it_comes_before_the_replayed_tool_images(self) -> None:
        channel = _channel()
        tool_image = to_data_uri(b"tool-image", "image/png")
        channel._tool_usage.record(  # noqa: SLF001
            "r1", "generate_image", {"prompt": "a lamp"}, [AIImagePart(url=tool_image)]
        )

        context = await _turn(channel, _message("compare with mine", "photo-1"))

        content = context.messages[-1].content
        assert isinstance(content, list)
        urls = [p.url for p in content if isinstance(p, AIImagePart)]
        assert urls == [_PNG, tool_image]
        header = next(
            i
            for i, p in enumerate(content)
            if isinstance(p, AITextPart) and p.text.startswith(TURN_IMAGES_HEADER)
        )
        assert content.index(next(p for p in content if isinstance(p, AIImagePart))) < header

    async def test_they_follow_the_events_own_media(self) -> None:
        own = MediaContent(url=_PNG, mime_type="image/png", filename="inline.png")
        event = _event(CompositeContent(parts=[TextContent(body="both"), own]))
        event = event.model_copy(update={"metadata": _files("photo-1")})
        host = _Host({_REF: b"attached"})

        context = await _turn(_channel(host=host), event)

        assert _images(context) == [_PNG, to_data_uri(b"attached", "image/png")]

    async def test_the_bound_counts_them_with_the_others(self) -> None:
        names = [f"p{i}" for i in range(5)]
        host = _Host({f"app-media://{n}": n.encode() for n in names})

        context = await _turn(_channel(host=host), _message("all of them", *names))

        assert _images(context) == [to_data_uri(n.encode(), "image/png") for n in names[1:]]
        assert "[Earlier image not shown: “p0.png”]" in _text(context)


class TestWithoutWords:
    async def test_a_model_that_sees_gets_the_image_alone(self) -> None:
        context = await _turn(_channel(), _message("", "photo-1"))

        assert context.messages[-1].content == [AIImagePart(url=_PNG, mime_type="image/png")]

    async def test_a_model_that_does_not_see_is_told_so(self) -> None:
        channel = _channel(vision=False, describe_empty_event=lambda event: "[a photo, no words]")

        context = await _turn(channel, _message("", "photo-1"))

        assert (
            context.messages[-1].content
            == "[Image not shown, this model does not see images: \u201cphoto-1.png\u201d]"
        )


class TestNotSeen:
    async def test_a_model_that_does_not_see_reads_a_mention(self, streaming: bool) -> None:
        host = _Host()

        context = await _turn(
            _channel(streaming, vision=False, host=host), _message("look", "photo-1")
        )

        assert (
            context.messages[-1].content
            == "look\n[Image not shown, this model does not see images: \u201cphoto-1.png\u201d]"
        )
        assert host.asked == []

    async def test_a_reader_that_fails_names_none(self, caplog: pytest.LogCaptureFixture) -> None:
        def broken(event: RoomEvent) -> list[MediaContent]:
            raise RuntimeError("record unreadable")

        with caplog.at_level(logging.WARNING, logger="roomkit.channels.ai"):
            context = await _turn(_channel(reader=broken), _message("look", "photo-1"))

        assert context.messages[-1].content == "look"
        assert "attached-media reader failed" in caplog.text

    async def test_an_instruction_is_read_as_it_is(self) -> None:
        asked: list[str] = []

        def reader(event: RoomEvent) -> list[MediaContent]:
            asked.append(event.id)
            return _reader(event)

        instruction = _message("summarize the room", "photo-1").model_copy(
            update={"type": EventType.INSTRUCTION}
        )

        context = await _turn(_channel(reader=reader), instruction)

        assert _images(context) == []
        assert instruction.id not in asked


class TestMemory:
    async def test_the_memory_reads_the_events_as_the_model_does(self) -> None:
        seen: list[RoomEvent] = []

        class Watching(MemoryProvider):
            async def retrieve(
                self,
                room_id: str,
                current_event: RoomEvent,
                context: RoomContext,
                *,
                channel_id: str | None = None,
            ) -> MemoryResult:
                seen.extend(context.recent_events)
                events = [e for e in context.recent_events if e.id != current_event.id]
                return MemoryResult(events=events)

        photo = _message("my lamp", "photo-1")

        await _turn(_channel(memory=Watching()), _event(TextContent(body="?")), [photo])

        read = next(e for e in seen if e.id == photo.id)
        assert isinstance(read.content, CompositeContent)
        assert read.content.parts[-1] == _reader(photo)[0]
        # What a budget measures: the image counts, as an inline one would.
        assert estimate_event_tokens(read) > estimate_event_tokens(photo)


class TestRefusal:
    async def test_an_attached_image_the_provider_refused_reads_as_refused_after(self) -> None:
        refused = ProviderError(
            "400 the image could not be processed", retryable=False, status_code=400
        )
        provider = _Scripted([refused, _OK])
        channel = _channel(provider=provider)
        photo = _message("my lamp", "photo-1")
        first = RoomContext(
            room=Room(id="r1"), bindings=[_MEMBER, _BINDING], recent_events=[photo]
        )
        with contextlib.suppress(ProviderError):
            await read_reply(await channel.on_event(photo, _BINDING, first))
        question = _event(TextContent(body="and now?"))

        context = await _turn(channel, question, [photo])

        assert _images(context) == []
        assert "[Image refused by the model's provider: “photo-1.png”]" in _text(context)
