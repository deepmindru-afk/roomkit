"""The default transcoding rules (RFC §6.6)."""

from __future__ import annotations

import pytest

from roomkit import RoomKit
from roomkit._media import media_label
from roomkit.channels import SMSChannel, TelegramChannel, WhatsAppPersonalChannel
from roomkit.channels.base import Channel
from roomkit.core.event_router import _cut_content
from roomkit.core.transcoder import DefaultContentTranscoder
from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import (
    AudioContent,
    CompositeContent,
    EventContent,
    LocationContent,
    MediaContent,
    TextContent,
    VideoContent,
)
from roomkit.providers.sms.mock import MockSMSProvider
from roomkit.providers.telegram.mock import MockTelegramProvider
from roomkit.providers.whatsapp.mock import MockWhatsAppProvider
from tests.test_framework import SimpleChannel


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


class _PhotoPhone(SimpleChannel):
    """A transport that takes JPEG files of 1000 bytes at most."""

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            media_types=[ChannelMediaType.TEXT, ChannelMediaType.MEDIA],
            supported_media_types=["image/jpeg"],
            max_media_size_bytes=1000,
        )


async def _delivered(target: Channel, content: EventContent) -> EventContent:
    """What *target* receives when another channel of its room sends *content*."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("in"))
    kit.register_channel(target)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "in")
    key = target.recipient_key
    await kit.attach_channel(
        "r1", target.channel_id, metadata={key: "+15550000001"} if key else {}
    )
    await kit.process_inbound(
        InboundMessage(channel_id="in", sender_id="u1", content=content), room_id="r1"
    )
    await kit.close()
    delivered = getattr(target, "delivered", None)
    if delivered is None:
        delivered = [sent["event"] for sent in target._provider.sent]  # noqa: SLF001
    assert len(delivered) == 1
    return delivered[0].content


_JPEG = "https://example.com/a.jpg"


class TestAFileTheTargetDoesNotTake:
    """RFC §5.8, §6.6: a file whose type a non-empty ``supported_media_types``
    does not list, or whose known size is over ``max_media_size_bytes``, is
    delivered as its text form, never dropped nor sent as is; an unknown
    size passes."""

    @pytest.mark.parametrize(
        ("media", "taken"),
        [
            (MediaContent(url=_JPEG, mime_type="image/jpeg"), True),
            (MediaContent(url=_JPEG, mime_type="image/jpeg", size_bytes=1000), True),
            (MediaContent(url=_JPEG, mime_type="image/jpeg", size_bytes=1001), False),
            (
                MediaContent(url="https://example.com/r.pdf", mime_type="application/pdf"),
                False,
            ),
        ],
        ids=["unknown-size", "at-the-limit", "over-the-limit", "unlisted-type"],
    )
    async def test_a_file_is_delivered_only_when_the_target_takes_it(
        self, media: MediaContent, taken: bool
    ) -> None:
        delivered = await _delivered(_PhotoPhone("out"), media)

        assert delivered == (media if taken else TextContent(body=media_label(media)))

    async def test_a_composites_untaken_file_becomes_its_text_and_no_part_is_dropped(
        self,
    ) -> None:
        receipt = MediaContent(
            url="https://example.com/r.pdf", mime_type="application/pdf", filename="r.pdf"
        )
        composite = CompositeContent(parts=[TextContent(body="Your receipt"), receipt])

        delivered = await _delivered(_PhotoPhone("out"), composite)

        assert delivered == TextContent(body="Your receipt\n[Media: r.pdf]")


class TestTheDeclarationsChannelsShip:
    """RFC Appendix A: what a channel lists decides what it is sent."""

    async def test_sms_takes_the_images_appendix_a_lists(self) -> None:
        pdf = MediaContent(url="https://example.com/r.pdf", mime_type="application/pdf")
        photo = MediaContent(url=_JPEG, mime_type="image/jpeg")

        assert await _delivered(SMSChannel("out", provider=MockSMSProvider()), pdf) == (
            TextContent(body=media_label(pdf))
        )
        assert await _delivered(SMSChannel("out", provider=MockSMSProvider()), photo) == photo

    async def test_whatsapp_personal_takes_a_document(self) -> None:
        pdf = MediaContent(url="https://example.com/r.pdf", mime_type="application/pdf")
        channel = WhatsAppPersonalChannel("out", provider=MockWhatsAppProvider())

        assert await _delivered(channel, pdf) == pdf

    @pytest.mark.parametrize(
        "content",
        [
            AudioContent(url="https://example.com/a.ogg"),
            VideoContent(url="https://example.com/v.mp4"),
        ],
        ids=["audio", "video"],
    )
    async def test_telegram_takes_audio_and_video(self, content: EventContent) -> None:
        channel = TelegramChannel("out", provider=MockTelegramProvider())

        assert await _delivered(channel, content) == content
