"""RoomKit.post() — the host's own voice in a room (RFC §10.5, RMK-947)."""

from __future__ import annotations

import pytest

from roomkit import (
    ChannelCategory,
    ChannelType,
    Discussion,
    EventDirection,
    EventType,
    HostChannel,
    Participant,
    RoomKit,
    SMSChannel,
)
from roomkit.channels.agent import Agent
from roomkit.core.exceptions import ChannelNotRegisteredError
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider

CUSTOMER = "+15550100001"


class _CountingHost(HostChannel):
    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.delivered = 0

    async def deliver(self, event, binding, context):
        self.delivered += 1
        return await super().deliver(event, binding, context)


@pytest.fixture
async def kit():
    kit = RoomKit()
    yield kit
    await kit.close()


async def _sms_room(
    kit: RoomKit, *, agent: MockAIProvider | None, **room: object
) -> MockSMSProvider:
    """Room "r": the customer on SMS, and an agent "assistant" when given a provider."""
    sms = MockSMSProvider()
    kit.register_channel(SMSChannel("sms", provider=sms))
    if agent is not None:
        kit.register_channel(Agent("assistant", provider=agent))
    await kit.create_room(room_id="r", **room)
    await kit.attach_channel("r", "sms", metadata={"phone_number": CUSTOMER})
    if agent is not None and "orchestration" not in room:
        await kit.attach_channel("r", "assistant", category=ChannelCategory.INTELLIGENCE)
    return sms


def _texted(sms: MockSMSProvider) -> list[tuple[str, str]]:
    return [(item["to"], item["event"].content.body) for item in sms.sent]


async def test_post_texts_the_correspondent_and_solicits_no_agent(kit):
    ai = MockAIProvider(["should not answer"])
    sms = await _sms_room(kit, agent=ai)

    event = await kit.post("r", "Your order shipped.")

    assert _texted(sms) == [(CUSTOMER, "Your order shipped.")]
    assert ai.calls == []
    assert (event.source.channel_type, event.source.direction) == (
        ChannelType.HOST,
        EventDirection.INBOUND,
    )


async def test_post_addressed_to_an_agent_is_answered_and_names_the_speaker(kit):
    ai = MockAIProvider(["It left the warehouse today."])
    sms = await _sms_room(kit, agent=ai)
    host = _CountingHost("desk-host")
    kit.register_channel(host)
    # Turns are labelled with their speaker once two people take part (RFC §6.4).
    await kit.store.add_participant(
        Participant(id=CUSTOMER, room_id="r", channel_id="sms", display_name="Marie")
    )
    await kit.store.add_participant(
        Participant(id="desk", room_id="r", channel_id="desk-host", display_name="Acme Support")
    )

    await kit.post(
        "r",
        "Tell the customer where the order is.",
        participant_id="desk",
        addressed_to=["assistant"],
        channel_id="desk-host",
    )

    asked = [m.content for m in ai.calls[0].messages if m.role == "user"]
    assert asked[-1].startswith('Acme Support: "Tell the customer')
    assert _texted(sms)[-1] == (CUSTOMER, "It left the warehouse today.")
    assert host.delivered == 0


async def test_post_attaches_the_host_channel_once(kit):
    await _sms_room(kit, agent=None)

    await kit.post("r", "One.")
    await kit.post("r", "Two.")

    events = await kit.store.list_events("r", limit=100)
    attached = [
        e for e in events if e.type == EventType.CHANNEL_ATTACHED and "host" in e.content.body
    ]
    assert len(attached) == 1
    assert await kit.store.get_binding("r", "host") is not None


async def test_post_reaches_the_correspondent_of_a_discussion_room(kit):
    sms = await _sms_room(
        kit, agent=MockAIProvider(["unused"]), orchestration=Discussion(agents=["assistant"])
    )

    await kit.post("r", "The meeting moved to 3 pm.")

    assert _texted(sms) == [(CUSTOMER, "The meeting moved to 3 pm.")]


@pytest.mark.parametrize(
    ("channel_id", "error"),
    [("sms", ValueError), ("ghost", ChannelNotRegisteredError)],
)
async def test_post_refuses_a_channel_that_is_not_a_host_before_writing(kit, channel_id, error):
    sms = await _sms_room(kit, agent=None)
    before = len(await kit.store.list_events("r", limit=100))

    with pytest.raises(error):
        await kit.post("r", "Never sent.", channel_id=channel_id)

    assert len(await kit.store.list_events("r", limit=100)) == before
    assert sms.sent == []
