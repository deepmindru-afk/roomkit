"""One event-text extraction, two fallbacks.

The transports read an event's text through
``roomkit.providers.utils.extract_event_text`` and send nothing when there is
none; the memory layer reads it through
``roomkit.memory.token_estimator.extract_event_text``, which builds on the same
extraction and renders text-less content instead, since it still costs tokens
and still belongs in a summary.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.memory.token_estimator import extract_event_text as memory_text
from roomkit.models.enums import ChannelType
from roomkit.models.event import (
    CompositeContent,
    EventSource,
    MediaContent,
    RichContent,
    RoomEvent,
    TemplateContent,
    TextContent,
)
from roomkit.providers.utils import extract_event_text as transport_text


def _event(content: Any) -> RoomEvent:
    return RoomEvent(
        room_id="room-1",
        source=EventSource(channel_id="ch1", channel_type=ChannelType.SMS),
        content=content,
    )


class TestTransportText:
    def test_text_body(self) -> None:
        assert transport_text(_event(TextContent(body="hello"))) == "hello"

    def test_rich_body(self) -> None:
        assert transport_text(_event(RichContent(body="**hi**"))) == "**hi**"

    def test_template_without_body_is_empty_not_none(self) -> None:
        assert transport_text(_event(TemplateContent(template_id="welcome"))) == ""

    def test_media_is_empty(self) -> None:
        content = MediaContent(url="https://x.example/a.png", mime_type="image/png")
        assert transport_text(_event(content)) == ""


class TestMemoryText:
    def test_text_body_even_when_empty(self) -> None:
        assert memory_text(_event(TextContent(body=""))) == ""

    def test_rich_body_is_the_body_not_the_repr(self) -> None:
        assert memory_text(_event(RichContent(body="**hi**"))) == "**hi**"

    def test_media_is_rendered_as_its_label(self) -> None:
        content = MediaContent(url="https://x.example/a.png", mime_type="image/png")
        assert memory_text(_event(content)) == "[Media: https://x.example/a.png]"

    @pytest.mark.parametrize(
        "url", ["data:image/png;base64,iVBORw0KGgo=", "app-media://tenant-a/secret/photo-1"]
    )
    def test_a_summary_never_reads_a_payload_or_a_reference(self, url: str) -> None:
        # A memory's summary quotes this line: neither base64 nor a tenant's
        # storage path belongs in it (RFC §6.6).
        text = memory_text(_event(MediaContent(url=url, mime_type="image/png")))
        assert text == "[Media: image/png]"

    def test_a_composite_reads_its_words_and_its_media(self) -> None:
        content = CompositeContent(
            parts=[
                TextContent(body="compare with mine"),
                MediaContent(url="app-media://p1", mime_type="image/png", filename="lamp.png"),
            ]
        )
        assert memory_text(_event(content)) == "compare with mine\n[Media: lamp.png]"

    def test_template_without_body_is_rendered(self) -> None:
        content = TemplateContent(template_id="welcome")
        assert memory_text(_event(content)) == str(content)
