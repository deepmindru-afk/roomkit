"""A message with a media, read outside the AI channel (RFC §6.6, §19.7.4, RMK-767).

Every reader that takes a room's message as text reads it the way the AI
channel does: its text, each media as its label, a composite's parts. One that
reads names in it, or a search query, reads the author's words alone, a
caption included, never a file name. A Loop answers a message that is not
plain text through its producer alone, which sees the media.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from roomkit import RoomKit
from roomkit._media import content_text, content_words
from roomkit.channels.agent import Agent
from roomkit.memory.retrieval import RetrievalMemory
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import (
    CompositeContent,
    EventContent,
    EventSource,
    MediaContent,
    RichContent,
    RoomEvent,
    TextContent,
)
from roomkit.models.room import Room
from roomkit.orchestration.strategies.discussion._driver import TurnDriver
from roomkit.orchestration.strategies.discussion.dispatch import classifier as dispatch
from roomkit.orchestration.strategies.loop import Loop
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.speaking import classifier as speaking
from roomkit.tasks.brief import conversation_block
from tests.orchestration.test_loop_strategy import _make_mock_kit
from tests.orchestration.test_strategy_discussion import _provider, _room, _settle
from tests.tool_loop_modes import read_reply

_PHOTO = MediaContent(
    url="app-media://tenant-a/photo-1", mime_type="image/png", filename="lamp.png"
)
_COMPOSITE = CompositeContent(parts=[TextContent(body="compare with mine"), _PHOTO])


def _event(content: EventContent, channel_id: str = "sms1") -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(channel_id=channel_id, channel_type=ChannelType.SMS),
        content=content,
    )


class TestSharedReadings:
    def test_the_whole_message_reads_its_text_and_its_media_label(self) -> None:
        assert content_text(_COMPOSITE) == "compare with mine\n[Media: lamp.png]"

    def test_the_authors_words_keep_a_caption_and_never_a_file_name(self) -> None:
        captioned = _PHOTO.model_copy(update={"caption": "my lamp", "filename": "@nova.png"})

        assert content_words(captioned) == "my lamp"
        assert content_words(_PHOTO) == ""
        assert content_words(_COMPOSITE) == "compare with mine"

    def test_rich_content_reads_as_its_plain_text(self) -> None:
        rich = RichContent(body="**hi**", plain_text="hi")

        assert content_text(rich) == content_words(rich) == "hi"


class TestLoop:
    @pytest.mark.parametrize("content", [_COMPOSITE, _PHOTO], ids=["composite", "photo"])
    async def test_a_message_with_a_photo_is_answered_by_the_producer(
        self, content: EventContent
    ) -> None:
        producer = MockAIProvider(responses=["a red lamp, like yours"], vision=True)
        reviewer = MockAIProvider(responses=["approved"])
        writer = Agent("writer", provider=producer)
        kit = _make_mock_kit(Room(id="r1"))
        await Loop(agent=writer, reviewers=[Agent("editor", provider=reviewer)]).install(kit, "r1")
        binding = ChannelBinding(
            channel_id="writer",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )

        output = await writer.on_event(_event(content), binding, RoomContext(room=Room(id="r1")))
        run = await read_reply(output)

        # Answered, where the loop rendered nothing; the producer saw the
        # message itself, no reviewer was asked.
        assert run.text == "a red lamp, like yours"
        assert len(producer.calls) == 1 and reviewer.calls == []


class TestTextReaders:
    def test_a_workers_brief_carries_a_composite_and_a_media(self) -> None:
        bindings = [
            ChannelBinding(channel_id="sms1", room_id="r1", channel_type=ChannelType.SMS),
            ChannelBinding(channel_id="worker", room_id="r1", channel_type=ChannelType.AI),
        ]
        context = RoomContext(
            room=Room(id="r1"),
            bindings=bindings,
            recent_events=[_event(_COMPOSITE), _event(_PHOTO)],
        )

        block = conversation_block(context, "worker", 5)

        assert block is not None
        assert "compare with mine\\n[Media: lamp.png]" in block  # a JSON string
        assert '"[Media: lamp.png]"' in block
        assert "app-media://" not in block

    @pytest.mark.parametrize("reader", [speaking._text, dispatch._text], ids=["speak", "dispatch"])
    def test_a_classifier_reads_a_composite(self, reader: object) -> None:
        assert reader(_event(_COMPOSITE)) == "compare with mine\n[Media: lamp.png]"  # type: ignore[operator]

    def test_a_discussion_turn_quotes_the_message_it_answers(self) -> None:
        stub = SimpleNamespace(
            _room=SimpleNamespace(agents={}, label=lambda channel_id: channel_id)
        )

        note = TurnDriver._answering(stub, _event(_COMPOSITE), RoomContext(room=Room(id="r1")))  # type: ignore[arg-type]

        assert "compare with mine [Media: lamp.png]" in note

    def test_a_search_query_reads_the_caption_never_the_file_name(self) -> None:
        captioned = _PHOTO.model_copy(update={"caption": "my red lamp"})

        assert RetrievalMemory._extract_query(_event(captioned)) == "my red lamp"
        assert RetrievalMemory._extract_query(_event(_COMPOSITE)) == "compare with mine"


class TestDiscussionAddressing:
    @staticmethod
    async def _say(kit: RoomKit, content: EventContent) -> None:
        await kit.process_inbound(
            InboundMessage(channel_id="ops", sender_id="oncall", content=content)
        )

    async def test_a_name_in_a_composite_asks_that_agent_alone(self) -> None:
        a, b = _provider("a answers"), _provider("b answers")
        kit = RoomKit()
        await _room(kit, {"a": a, "b": b})

        await self._say(kit, CompositeContent(parts=[TextContent(body="@a look at this"), _PHOTO]))
        await _settle(kit)

        assert (len(a.calls), len(b.calls)) == (1, 0)
        await kit.close()

    async def test_a_name_in_a_file_name_addresses_no_one(self) -> None:
        a, b = _provider("a answers"), _provider("b answers")
        kit = RoomKit()
        await _room(kit, {"a": a, "b": b})
        named_file = _PHOTO.model_copy(update={"filename": "@b notes.png"})

        await self._say(
            kit, CompositeContent(parts=[TextContent(body="what is this?"), named_file])
        )
        await _settle(kit)

        # Unaddressed: everyone is asked, b is not singled out by the file name.
        assert (len(a.calls), len(b.calls)) == (1, 1)
        await kit.close()
