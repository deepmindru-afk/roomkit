"""The hooks of a turn register on the turn executor (RFC §10.6.6): the kit
refuses them, an executor refuses the owner's, and the kit's own executor
fires them for the turns it runs."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.core.hooks import TURN_TRIGGERS, HookRegistration
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, HookExecution, HookTrigger
from roomkit.models.event import TextContent
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


async def _noop(event: Any, ctx: Any) -> None:
    return None


@pytest.mark.parametrize("trigger", sorted(TURN_TRIGGERS, key=str))
def test_the_kit_refuses_a_turn_trigger_on_every_door(trigger: HookTrigger) -> None:
    kit = RoomKit()
    with pytest.raises(ValueError, match=r"kit\.turn_executor\.hook"):
        kit.hook(trigger, execution=HookExecution.ASYNC)(_noop)
    with pytest.raises(ValueError, match=r"kit\.turn_executor\.hook"):
        kit.add_room_hook("room", trigger, HookExecution.ASYNC, _noop)
    with pytest.raises(ValueError, match=r"kit\.turn_executor\.hook"):
        kit.hook_engine.register(
            HookRegistration(trigger=trigger, execution=HookExecution.ASYNC, fn=_noop)
        )


@pytest.mark.parametrize(
    "trigger", [HookTrigger.BEFORE_BROADCAST, HookTrigger.AFTER_BROADCAST, HookTrigger.ON_ERROR]
)
def test_an_executor_refuses_an_owners_trigger(trigger: HookTrigger) -> None:
    kit = RoomKit()
    with pytest.raises(ValueError, match=r"kit\.hook\("):
        kit.turn_executor.hook(trigger, execution=HookExecution.ASYNC)(_noop)


async def _room(kit: RoomKit, room_id: str) -> None:
    await kit.create_room(room_id=room_id)
    await kit.attach_channel(room_id, "sms")
    await kit.attach_channel(room_id, "ai", category=ChannelCategory.INTELLIGENCE)


async def _ask(kit: RoomKit, room_id: str) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="+15550000001", content=TextContent(body="hi")),
        room_id=room_id,
    )


async def test_a_hook_on_the_kits_executor_fires_for_an_ai_turn() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["hello"])))
    await _room(kit, "room")
    seen: list[str] = []

    @kit.turn_executor.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def on_response(event: Any, ctx: Any) -> None:
        seen.append(event.channel_id)

    await _ask(kit, "room")
    await kit.close()
    assert seen == ["ai"]


async def test_a_room_hook_on_the_executor_fires_in_its_room_only() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["a", "b"])))
    await _room(kit, "one")
    await _room(kit, "two")
    seen: list[str] = []

    async def on_response(event: Any, ctx: Any) -> None:
        seen.append(ctx.room.id)

    kit.turn_executor.add_room_hook(
        "one", HookTrigger.ON_AI_RESPONSE, HookExecution.ASYNC, on_response
    )
    await _ask(kit, "one")
    await _ask(kit, "two")
    await kit.close()
    assert seen == ["one"]
