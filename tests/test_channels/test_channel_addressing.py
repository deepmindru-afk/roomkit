"""A channel's type decides how its addresses are written and where its
recipient and its chat are (RFC §5.7, Appendix A)."""

from __future__ import annotations

import pytest

from roomkit import RoomKit
from roomkit.channels import TransportChannel, VoiceChannel
from roomkit.channels.addressing import Addressing, ChannelAddressing, addressing_of
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType
from roomkit.models.event import TextContent
from roomkit.providers.sms.mock import MockSMSProvider
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

PHONE = ChannelAddressing(
    Addressing.PHONE_NUMBER, recipient_key="phone_number", replies_to_sender=True
)


@pytest.mark.parametrize(
    ("channel_type", "expected"),
    [
        (ChannelType.SMS, PHONE),
        (ChannelType.MMS, PHONE),
        (ChannelType.RCS, PHONE),
        (ChannelType.WHATSAPP, PHONE),
        (
            ChannelType.WHATSAPP_PERSONAL,
            ChannelAddressing(
                Addressing.PHONE_NUMBER,
                recipient_key="phone_number",
                chat_key="group_jid",
                replies_to_sender=True,
            ),
        ),
        (ChannelType.VOICE, ChannelAddressing(Addressing.PHONE_NUMBER)),
        (
            ChannelType.EMAIL,
            ChannelAddressing(
                Addressing.EMAIL_ADDRESS, recipient_key="email_address", replies_to_sender=True
            ),
        ),
        (
            ChannelType.MESSENGER,
            ChannelAddressing(
                Addressing.AS_WRITTEN, recipient_key="facebook_user_id", replies_to_sender=True
            ),
        ),
        (
            ChannelType.TELEGRAM,
            ChannelAddressing(
                Addressing.CHAT, recipient_key="telegram_chat_id", chat_key="chat_id"
            ),
        ),
        (
            ChannelType.TEAMS,
            ChannelAddressing(
                Addressing.CHAT, recipient_key="teams_conversation_id", chat_key="conversation_id"
            ),
        ),
        (ChannelType.WEBSOCKET, ChannelAddressing(Addressing.AS_WRITTEN)),
        (ChannelType.WEBHOOK, ChannelAddressing(Addressing.AS_WRITTEN)),
        (ChannelType.REALTIME_VOICE, ChannelAddressing(Addressing.AS_WRITTEN)),
    ],
)
def test_the_appendix_a_table(channel_type: ChannelType, expected: ChannelAddressing) -> None:
    assert addressing_of(channel_type) == expected


@pytest.mark.parametrize(
    ("addressing", "names_conversations"),
    [
        (PHONE, True),
        (addressing_of(ChannelType.WHATSAPP_PERSONAL), True),
        (addressing_of(ChannelType.TELEGRAM), True),
        (addressing_of(ChannelType.VOICE), False),
        (ChannelAddressing(Addressing.AS_WRITTEN, recipient_key="recipient_id"), False),
    ],
    ids=["phone", "whatsapp-personal", "telegram", "voice", "forwarded"],
)
def test_a_recipient_names_a_conversation_when_replies_go_to_the_sender_or_a_chat(
    addressing: ChannelAddressing, names_conversations: bool
) -> None:
    assert addressing.names_conversations is names_conversations


@pytest.mark.parametrize("channel_type", [ChannelType.SMS, ChannelType.MMS])
async def test_a_transport_built_by_type_takes_its_types_addressing(
    channel_type: ChannelType,
) -> None:
    """A transport channel built for its type, without a factory: its
    senders in E.164, recorded as its recipient under ``phone_number``."""
    kit = RoomKit()
    kit.register_channel(TransportChannel("t", channel_type, provider=MockSMSProvider()))

    result = await kit.process_inbound(
        InboundMessage(
            channel_id="t", sender_id="+1 (555) 000-0001", content=TextContent(body="hi")
        )
    )

    assert result.event is not None
    assert result.event.source.participant_id == "+15550000001"
    binding = await kit.store.get_binding(result.event.room_id, "t")
    assert binding is not None
    assert (binding.participant_id, binding.metadata) == (
        "+15550000001",
        {"phone_number": "+15550000001"},
    )
    await kit.close()


def test_a_voice_channel_writes_numbers_in_e164() -> None:
    voice = VoiceChannel(
        "voice", stt=MockSTTProvider(), tts=MockTTSProvider(), backend=MockVoiceBackend()
    )

    assert voice.normalize_address("+1 (555) 000-0001") == "+15550000001"
    assert voice.normalize_address("caller-7") == "caller-7"
