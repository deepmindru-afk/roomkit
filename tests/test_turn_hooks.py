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
from roomkit.orchestration.base import Orchestration
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


async def _noop(event: Any, ctx: Any) -> None:
    return None


# RFC §10.6.6's table, written out: a trigger dropped from the set is caught.
_RFC_TURN_TRIGGERS = {
    "before_ai_generation",
    "on_ai_thinking",
    "on_ai_response",
    "before_tool_use",
    "after_tool_round",
    "on_tool_call",
    "on_user_input_required",
    "on_speak_decision",
    "on_thought",
    "on_plan_updated",
}
_OWNER_TRIGGERS = sorted(set(HookTrigger) - TURN_TRIGGERS, key=str)


def test_the_turn_triggers_are_the_rfcs() -> None:
    assert {str(t) for t in TURN_TRIGGERS} == _RFC_TURN_TRIGGERS


@pytest.mark.parametrize("trigger", sorted(TURN_TRIGGERS, key=str))
def test_the_kit_refuses_a_turn_trigger_on_every_door(trigger: HookTrigger) -> None:
    kit = RoomKit()
    with pytest.raises(ValueError, match=r"kit\.turn_executor\.hook\(\.\.\.\)"):
        kit.hook(trigger, execution=HookExecution.ASYNC)(_noop)
    with pytest.raises(ValueError, match=r"kit\.turn_executor\.add_room_hook\(room_id"):
        kit.add_room_hook("room", trigger, HookExecution.ASYNC, _noop)
    with pytest.raises(ValueError, match=r"kit\.turn_executor\.hooks\.register"):
        kit.hook_engine.register(
            HookRegistration(trigger=trigger, execution=HookExecution.ASYNC, fn=_noop)
        )


@pytest.mark.parametrize("trigger", _OWNER_TRIGGERS)
def test_an_executor_refuses_an_owners_trigger_on_every_door(trigger: HookTrigger) -> None:
    executor = RoomKit().turn_executor
    with pytest.raises(ValueError, match=r"kit\.hook\(\.\.\.\)"):
        executor.hook(trigger, execution=HookExecution.ASYNC)(_noop)
    with pytest.raises(ValueError, match=r"kit\.add_room_hook\(room_id"):
        executor.add_room_hook("room", trigger, HookExecution.ASYNC, _noop)


def test_each_side_fails_closed_on_its_own_triggers() -> None:
    kit = RoomKit()
    assert HookTrigger.BEFORE_TOOL_USE in kit.turn_executor.hooks.FAIL_CLOSED_TRIGGERS
    assert HookTrigger.BEFORE_TTS in kit.hook_engine.FAIL_CLOSED_TRIGGERS
    with pytest.raises(ValueError, match=r"kit\.turn_executor\.hooks\.FAIL_CLOSED_TRIGGERS"):
        kit.hook_engine.FAIL_CLOSED_TRIGGERS = frozenset({HookTrigger.ON_TOOL_CALL})
    with pytest.raises(ValueError, match=r"kit\.hook_engine\.FAIL_CLOSED_TRIGGERS"):
        kit.turn_executor.hooks.FAIL_CLOSED_TRIGGERS = frozenset({HookTrigger.BEFORE_BROADCAST})
    kit.turn_executor.hooks.FAIL_CLOSED_TRIGGERS = frozenset({HookTrigger.ON_TOOL_CALL})
    assert kit.turn_executor.hooks.fail_closed_hook("r", HookTrigger.ON_TOOL_CALL) is None


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


class _AuditStrategy(Orchestration):
    """A host strategy whose install puts a room hook on the turn executor."""

    def agents(self) -> list[Any]:
        return []

    async def install(self, kit: RoomKit, room_id: str) -> None:
        kit.turn_executor.add_room_hook(
            room_id, HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC, _noop, name="audit"
        )


async def test_uninstalling_a_strategy_takes_back_its_room_hook_on_the_executor() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="room")
    await kit.install_strategy("room", _AuditStrategy())
    assert kit.turn_executor.hooks.room_hook_names("room") == ["audit"]
    await kit.uninstall_strategy("room")
    assert kit.turn_executor.hooks.room_hook_names("room") == []
    await kit.close()
