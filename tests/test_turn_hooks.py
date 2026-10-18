"""The hooks of a turn fire on the turn executor (RFC §10.6.6): registered on
the kit they are its own executor's, refused only where a channel the kit
holds by its declaration alone could fire them; an executor refuses the
owner's, and the kit's own executor fires them for the turns it runs."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import AgentIdentity, ChannelDeclaration, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.channels.declared import DeclaredChannel, register_strategy_agents
from roomkit.core.hooks import TURN_TRIGGERS, HookEngine, HookRegistration
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
def test_a_turn_trigger_on_the_kit_is_its_own_executors_on_every_door(
    trigger: HookTrigger,
) -> None:
    kit = RoomKit()
    kit.hook(trigger, execution=HookExecution.ASYNC, name="global")(_noop)
    kit.add_room_hook("room", trigger, HookExecution.ASYNC, _noop, name="room")
    kit.hook_engine.register(
        HookRegistration(trigger=trigger, execution=HookExecution.ASYNC, fn=_noop, name="raw")
    )

    executor = kit.turn_executor.hooks
    assert [h.name for h in executor._get_hooks("room", trigger, None)] == [
        "global",
        "raw",
        "room",
    ]
    assert not kit.hook_engine._get_hooks("room", trigger, None)
    assert kit.hook_engine.room_hook_names("room") == ["room"]
    assert kit.remove_room_hook("room", "room")
    assert kit.hook_engine.remove_global_hook("global")
    assert [h.name for h in executor._get_hooks("room", trigger, None)] == ["raw"]


def _declared(kit: RoomKit, channel_id: str = "meteo") -> None:
    kit.register_channel(ChannelDeclaration(channel_id=channel_id, identity=AgentIdentity()))


@pytest.mark.parametrize("trigger", sorted(TURN_TRIGGERS, key=str))
def test_a_kit_turn_hook_a_declared_channel_could_fire_is_refused_in_either_order(
    trigger: HookTrigger,
) -> None:
    """Its turns run on another executor, where the hook would never fire."""
    declared_first = RoomKit()
    _declared(declared_first)
    with pytest.raises(ValueError, match=r"'meteo'.*executor\.hook\(\.\.\.\)"):
        declared_first.hook(trigger, execution=HookExecution.ASYNC)(_noop)
    with pytest.raises(ValueError, match=r"'meteo'"):
        declared_first.add_room_hook("room", trigger, HookExecution.ASYNC, _noop)

    hooked_first = RoomKit()
    hooked_first.hook(trigger, execution=HookExecution.ASYNC, name="audit")(_noop)
    with pytest.raises(ValueError, match=r"'meteo'.*'audit'"):
        _declared(hooked_first)
    assert "meteo" not in hooked_first._channels


def test_a_kit_turn_hook_whose_filter_leaves_the_declared_channel_out_is_kept() -> None:
    kit = RoomKit()
    _declared(kit, "meteo")
    seen: list[str] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, channel_ids={"support"})
    async def audit(event: Any, ctx: Any) -> None:
        seen.append(event.channel_id)

    _declared(kit, "billing")  # left out too

    with pytest.raises(ValueError, match=r"'support'"):
        kit.register_channel(ChannelDeclaration(channel_id="support", identity=AgentIdentity()))
    assert [h.name for h in kit.turn_executor.hooks._global_hooks] == ["audit"]


def test_a_kit_room_turn_hook_is_refused_while_any_channel_is_declared() -> None:
    """The room's channels are known only later (RFC §10.6.6)."""
    kit = RoomKit()
    kit.add_room_hook("room", HookTrigger.BEFORE_TOOL_USE, HookExecution.SYNC, _noop, name="gate")
    with pytest.raises(ValueError, match=r"'gate'.*for a room"):
        _declared(kit)


@pytest.mark.parametrize("trigger", _OWNER_TRIGGERS)
def test_an_executor_refuses_an_owners_trigger_on_every_door(trigger: HookTrigger) -> None:
    executor = RoomKit().turn_executor
    with pytest.raises(ValueError, match=r"kit\.hook\(\.\.\.\)"):
        executor.hook(trigger, execution=HookExecution.ASYNC)(_noop)
    with pytest.raises(ValueError, match=r"kit\.add_room_hook\(room_id"):
        executor.room_hook("audit", trigger, HookExecution.ASYNC)(_noop)


def test_the_kit_sets_the_fail_closed_triggers_of_both_sides() -> None:
    kit = RoomKit()
    assert HookTrigger.BEFORE_TOOL_USE in kit.turn_executor.hooks.FAIL_CLOSED_TRIGGERS
    assert HookTrigger.BEFORE_TOOL_USE in kit.hook_engine.FAIL_CLOSED_TRIGGERS
    assert HookTrigger.BEFORE_TTS in kit.hook_engine.FAIL_CLOSED_TRIGGERS

    kit.hook_engine.FAIL_CLOSED_TRIGGERS = frozenset(
        {HookTrigger.ON_TOOL_CALL, HookTrigger.BEFORE_BROADCAST}
    )

    assert {HookTrigger.ON_TOOL_CALL} == kit.turn_executor.hooks.FAIL_CLOSED_TRIGGERS
    assert {
        HookTrigger.ON_TOOL_CALL,
        HookTrigger.BEFORE_BROADCAST,
    } == kit.hook_engine.FAIL_CLOSED_TRIGGERS
    with pytest.raises(ValueError, match=r"kit\.hook_engine\.FAIL_CLOSED_TRIGGERS"):
        kit.turn_executor.hooks.FAIL_CLOSED_TRIGGERS = frozenset({HookTrigger.BEFORE_BROADCAST})


def test_a_bare_engine_still_refuses_a_turn_trigger() -> None:
    with pytest.raises(ValueError, match=r"fires on the turn executor"):
        HookEngine().register(
            HookRegistration(
                trigger=HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, fn=_noop
            )
        )


async def _room(kit: RoomKit, room_id: str) -> None:
    await kit.create_room(room_id=room_id)
    await kit.attach_channel(room_id, "sms")
    await kit.attach_channel(room_id, "ai", category=ChannelCategory.INTELLIGENCE)


async def _ask(kit: RoomKit, room_id: str) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="+15550000001", content=TextContent(body="hi")),
        room_id=room_id,
    )


async def test_a_turn_hook_on_the_kit_fires_for_an_ai_turn() -> None:
    """As in 0.105.0: ``@kit.hook`` on a trigger a turn fires."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["hello"])))
    await _room(kit, "room")
    seen: list[str] = []

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def on_response(event: Any, ctx: Any) -> None:
        seen.append(event.channel_id)

    await _ask(kit, "room")
    await kit.close()
    assert seen == ["ai"]


async def test_a_room_turn_hook_on_the_kit_fires_in_its_room_only() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["a", "b"])))
    await _room(kit, "one")
    await _room(kit, "two")
    seen: list[str] = []

    async def on_response(event: Any, ctx: Any) -> None:
        seen.append(ctx.room.id)

    kit.add_room_hook("one", HookTrigger.ON_AI_RESPONSE, HookExecution.ASYNC, on_response)
    await _ask(kit, "one")
    await _ask(kit, "two")
    await kit.close()
    assert seen == ["one"]


class _KitAuditStrategy(Orchestration):
    """A host strategy whose install puts a turn room hook on the kit."""

    def agents(self) -> list[Any]:
        return []

    async def install(self, kit: RoomKit, room_id: str) -> None:
        kit.add_room_hook(
            room_id, HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC, _noop, name="audit"
        )


async def test_uninstalling_a_strategy_takes_back_its_turn_room_hook_on_the_kit() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="room")
    await kit.install_strategy("room", _KitAuditStrategy())
    assert kit.hook_engine.room_hook_names("room") == ["audit"]
    await kit.uninstall_strategy("room")
    assert kit.hook_engine.room_hook_names("room") == []
    assert not kit.turn_executor.hooks.has_hooks(HookTrigger.ON_TOOL_CALL)
    await kit.close()


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

    kit.turn_executor.room_hook("audit", HookTrigger.ON_AI_RESPONSE, HookExecution.ASYNC)(
        on_response
    )
    kit.enable_room_hook("one", "audit")
    await _ask(kit, "one")
    await _ask(kit, "two")
    await kit.close()
    assert seen == ["one"]


class _AuditStrategy(Orchestration):
    """A host strategy whose install puts a room hook on the turn executor."""

    def agents(self) -> list[Any]:
        return []

    async def install(self, kit: RoomKit, room_id: str) -> None:
        kit.turn_executor.room_hook("audit", HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC)(_noop)
        kit.enable_room_hook(room_id, "audit")


async def test_uninstalling_a_strategy_takes_back_its_room_hook_on_the_executor() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="room")
    await kit.install_strategy("room", _AuditStrategy())
    assert kit._turns.room_hook_names("room") == ["audit"]
    assert kit.turn_executor.hooks.room_hook_names("room") == ["audit"]
    await kit.uninstall_strategy("room")
    assert kit._turns.room_hook_names("room") == []
    # Off the kit's executor at once, not at the room's next turn.
    assert kit.turn_executor.hooks.room_hook_names("room") == []
    await kit.close()


def test_a_kit_turn_hook_removed_on_the_executor_no_longer_refuses_a_declaration() -> None:
    kit = RoomKit()
    kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")(_noop)
    assert kit.turn_executor.hooks.remove_global_hook("audit")

    _declared(kit)

    assert "meteo" in kit._channels


async def test_a_turn_hook_s_channel_filter_holds_on_the_turn_s_agent() -> None:
    """RFC §9.1: a hook with channel_ids fires only for those channels' turns."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["a"])))
    kit.register_channel(AIChannel("other", provider=MockAIProvider(responses=["b"])))
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "sms")
    for agent in ("ai", "other"):
        await kit.attach_channel("room", agent, category=ChannelCategory.INTELLIGENCE)
    seen: list[str] = []

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC, channel_ids={"ai"})
    async def on_response(event: Any, ctx: Any) -> None:
        seen.append(event.channel_id)

    await _ask(kit, "room")
    await kit.close()
    # Both agents answered (each other too, up to the chain depth).
    assert seen and set(seen) == {"ai"}


def test_the_kit_s_engine_reads_the_turn_hooks_registered_through_it() -> None:
    """As in 0.105.0, where one engine held them all."""
    kit = RoomKit()
    kit.hook(HookTrigger.BEFORE_TOOL_USE, name="gate", fail_closed=True)(_noop)

    assert kit.hook_engine.has_hooks(HookTrigger.BEFORE_TOOL_USE)
    assert kit.hook_engine.has_hooks()
    assert kit.hook_engine.fail_closed_hook("room", HookTrigger.BEFORE_TOOL_USE) == "gate"


def test_one_registration_given_twice_is_removed_once_per_call() -> None:
    kit = RoomKit()
    twice = HookRegistration(
        trigger=HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, fn=_noop, name="dup"
    )
    kit.hook_engine.register(twice)
    kit.hook_engine.register(twice)

    assert kit.hook_engine.remove_global_hook("dup")
    assert [h.name for h in kit.turn_executor.hooks._global_hooks] == ["dup"]


def test_a_strategy_s_agents_are_registered_all_or_none_beside_a_kit_turn_hook() -> None:
    kit = RoomKit()
    kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)(_noop)
    live = AIChannel("live", provider=MockAIProvider(responses=["a"]))
    declared = DeclaredChannel(ChannelDeclaration(channel_id="meteo", identity=AgentIdentity()))

    with pytest.raises(ValueError, match=r"'meteo'"):
        register_strategy_agents(kit, [live, declared])

    assert "live" not in kit._channels and "meteo" not in kit._channels
