"""The default transcoding rules (RFC §6.6)."""

from __future__ import annotations

from roomkit.core.event_router import _cut_content
from roomkit.core.transcoder import DefaultContentTranscoder
from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import CompositeContent, LocationContent, MediaContent, TextContent


def _cut(body: str, max_length: int) -> str:
    cut = _cut_content(TextContent(body=body), max_length)
    assert isinstance(cut, TextContent)
    return cut.body


class TestMaxLength:
    def test_a_cut_text_ends_with_the_marker_within_the_limit(self) -> None:
        assert _cut("élèves présents", 6) == "élè..."

    def test_a_limit_below_four_cuts_without_the_marker(self) -> None:
        assert _cut("élève!", 3) == "élè"

    def test_a_text_within_the_limit_is_the_same_object(self) -> None:
        text = TextContent(body="short")
        assert _cut_content(text, 10) is text

    def test_each_text_part_of_a_composite_is_cut(self) -> None:
        photo = MediaContent(url="https://example.com/a.jpg", mime_type="image/jpeg")
        composite = CompositeContent(parts=[TextContent(body="Receipt attached"), photo])
        cut = _cut_content(composite, 7)
        assert isinstance(cut, CompositeContent)
        assert cut.parts == [TextContent(body="Rece..."), photo]


def _text_only(channel_id: str) -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id="r1",
        channel_type=ChannelType.SMS,
        capabilities=ChannelCapabilities(media_types=[ChannelMediaType.TEXT]),
    )


class TestLocation:
    async def test_a_location_without_a_name_reads_its_coordinates(self) -> None:
        transcoded = await DefaultContentTranscoder().transcode(
            LocationContent(latitude=45.5, longitude=-73.6), _text_only("a"), _text_only("b")
        )
        assert transcoded == TextContent(body="[Location: (45.5, -73.6)]")

    async def test_a_named_location_reads_its_name_first(self) -> None:
        transcoded = await DefaultContentTranscoder().transcode(
            LocationContent(latitude=45.5, longitude=-73.6, label="Office"),
            _text_only("a"),
            _text_only("b"),
        )
        assert transcoded == TextContent(body="[Location: Office (45.5, -73.6)]")
