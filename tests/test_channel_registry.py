"""The tools a channel serves, for every room or one (RFC §19.7, §21.1; RMK-307).

One channel object serves every room it is attached to. What orchestration sets
up for one room is served in that room only, and a name is served by one tool
in a room: a second one is refused when it is given.
"""

from __future__ import annotations

import contextvars

import pytest

from roomkit import ToolNameCollisionError
from roomkit.channels._tool_registry import (
    ChannelRegistry,
    ToolSource,
    TurnOwnerTools,
    channel_tool,
    orchestration_tool,
    set_turn_owner_tools,
)
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core._turn_runners import TurnRunners
from roomkit.providers.ai.base import AITool
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport


def _tool(name: str, description: str = "") -> AITool:
    return AITool(name=name, description=description or name, parameters={"type": "object"})


async def _serve(arguments: dict[str, object]) -> str:
    return "served"


def _registry(*host: str) -> ChannelRegistry:
    return ChannelRegistry("ch", lambda: host)


class TestScopes:
    def test_a_room_s_tool_is_served_in_that_room_only(self) -> None:
        registry = _registry()
        registry.register(orchestration_tool(_tool("delegate"), _serve), room_id="A", owner="s1")

        assert registry.lookup("delegate", "A") is not None
        assert registry.lookup("delegate", "B") is None
        assert registry.lookup("delegate", None) is None
        assert [e.name for e in registry.entries("B")] == []

    def test_two_rooms_serve_the_same_name_each_with_its_own(self) -> None:
        registry = _registry()
        in_a = orchestration_tool(_tool("delegate", "workers of A"), _serve)
        in_b = orchestration_tool(_tool("delegate", "workers of B"), _serve)
        registry.register(in_a, room_id="A", owner="s1")
        registry.register(in_b, room_id="B", owner="s2")

        assert registry.lookup("delegate", "A") is in_a
        assert registry.lookup("delegate", "B") is in_b

    def test_the_channel_s_tools_are_served_in_every_room_before_the_room_s(self) -> None:
        registry = _registry()
        registry.register(channel_tool(_tool("find_tools"), _serve), owner="ch")
        registry.register(orchestration_tool(_tool("delegate"), _serve), room_id="A", owner="s")

        assert registry.lookup("find_tools", "A") is not None
        assert [e.name for e in registry.entries("A")] == ["find_tools", "delegate"]
        assert [e.name for e in registry.entries("A", source=ToolSource.ORCHESTRATION)] == [
            "delegate"
        ]

    def test_traits_travel_with_the_entry(self) -> None:
        registry = _registry()
        registry.register(channel_tool(_tool("find_tools"), _serve), owner="ch")
        registry.register(orchestration_tool(_tool("delegate"), _serve), room_id="A", owner="s")

        find = registry.traits("find_tools")
        assert find is not None and find.exempt and find.pure and not find.in_digest
        assert registry.names("A", lambda traits: traits.always_declared) == {"delegate"}
        assert registry.names("A", lambda traits: not traits.deferrable) == {
            "find_tools",
            "delegate",
        }


class TestTurnOwnerTools:
    """Inside a turn the backend runs, the owner's tools are the request's
    (RFC §10.6.5); the owner reads its own entries."""

    def test_a_turn_declares_and_serves_the_owner_tools_its_request_lists(self) -> None:
        registry = _registry()
        own = channel_tool(_tool("plan_tasks"), _serve)
        registry.register(own)
        held = orchestration_tool(_tool("delegate"), _serve)
        registry.register(held, room_id="A", owner="s1")
        listed = orchestration_tool(_tool("delegate", "as the request lists it"), _serve)
        turn = TurnOwnerTools(channel_id="ch", room_id="A", entries={"delegate": listed})
        other = ChannelRegistry("other", lambda: ())
        others = orchestration_tool(_tool("delegate", "another channel's"), _serve)
        other.register(others, room_id="A", owner="s2")

        def in_turn() -> None:
            set_turn_owner_tools(turn)
            assert registry.lookup("delegate", "A") is listed
            assert registry.lookup("plan_tasks", "A") is own
            assert registry.entries("A") == [own, listed]
            # The owner's own reads are the registry's.
            assert registry.owner_entries("A") == [held]
            assert registry.owner_lookup("delegate", "A") is held
            # Another room, or another channel, is not this turn's.
            assert registry.lookup("delegate", "B") is None
            assert other.lookup("delegate", "A") is others

        contextvars.copy_context().run(in_turn)
        assert registry.lookup("delegate", "A") is held

    def test_a_turn_without_owner_tools_declares_none_of_the_registry_s(self) -> None:
        registry = _registry()
        registry.register(orchestration_tool(_tool("delegate"), _serve), owner="s1")
        turn = TurnOwnerTools(channel_id="ch", room_id="A", entries={})

        def in_turn() -> None:
            set_turn_owner_tools(turn)
            assert registry.lookup("delegate", "A") is None
            assert registry.entries("A") == []

        contextvars.copy_context().run(in_turn)


class TestCollisions:
    def test_a_second_tool_under_a_name_in_the_same_room_is_refused(self) -> None:
        registry = _registry()
        registry.register(orchestration_tool(_tool("delegate"), _serve), room_id="A", owner="s1")

        with pytest.raises(ToolNameCollisionError, match="'delegate'"):
            registry.register(
                orchestration_tool(_tool("delegate"), _serve), room_id="A", owner="s2"
            )

    def test_a_room_tool_under_a_channel_wide_name_is_refused_and_back(self) -> None:
        registry = _registry()
        registry.register(orchestration_tool(_tool("handoff"), _serve), owner="h")
        with pytest.raises(ToolNameCollisionError):
            registry.register(orchestration_tool(_tool("handoff"), _serve), room_id="A", owner="s")

        registry.register(orchestration_tool(_tool("delegate"), _serve), room_id="A", owner="s")
        with pytest.raises(ToolNameCollisionError, match="room 'A'"):
            registry.register(orchestration_tool(_tool("delegate"), _serve), owner="d")

    def test_an_orchestration_tool_under_a_host_tool_s_name_is_refused(self) -> None:
        registry = _registry("delegate_task")

        with pytest.raises(ToolNameCollisionError, match="a tool of the host"):
            registry.register(orchestration_tool(_tool("delegate_task"), _serve), owner="d")

    def test_the_same_owner_replaces_its_own_entry(self) -> None:
        registry = _registry()
        first = orchestration_tool(_tool("delegate", "first"), _serve)
        again = orchestration_tool(_tool("delegate", "again"), _serve)
        registry.register(first, room_id="A", owner="s")
        registry.register(again, room_id="A", owner="s")

        assert registry.lookup("delegate", "A") is again

    def test_only_its_owner_withdraws_an_entry(self) -> None:
        registry = _registry()
        registry.register(orchestration_tool(_tool("submit"), _serve), room_id="A", owner="c1")

        registry.unregister("submit", room_id="A", owner="c2")
        assert registry.lookup("submit", "A") is not None
        registry.unregister("submit", room_id="A", owner="c1")
        assert registry.lookup("submit", "A") is None


class TestInstalls:
    async def test_a_room_s_turns_are_taken_by_its_runner_only(self) -> None:
        # The owner keeps the runners, per channel and room (RFC §10.6.1).
        runners = TurnRunners()

        async def runner(*args: object) -> None:
            return None

        runners.take("A", "agent", runner, owner="loop")  # ty: ignore[invalid-argument-type]
        assert runners.get("A", "agent") is runner
        assert runners.get("B", "agent") is None
        assert runners.get("A", "other-agent") is None
        with pytest.raises(ValueError, match="another strategy"):
            runners.take("A", "agent", runner, owner="other")  # ty: ignore[invalid-argument-type]
        runners.release("A", "agent")
        assert runners.channels("A") == set()


class TestHostTools:
    def test_a_host_tool_given_twice_is_refused(self) -> None:
        """Two tool servers exposing one name: declared once and served by the
        other, the model would call one schema on the other's server."""
        with pytest.raises(ValueError, match="'search' is given twice"):
            AIChannel(
                "ai",
                provider=MockAIProvider(responses=["ok"]),
                tools=[_tool("search", "server A"), _tool("search", "server B")],
            )

    def test_the_channel_s_own_tools_are_entries_with_their_traits(self) -> None:
        channel = AIChannel("ai", provider=MockAIProvider(responses=["ok"]), enable_planning=True)

        own = {e.name for e in channel._registry.entries(None, source=ToolSource.CHANNEL)}
        assert own == {"read_stored_result", "plan_tasks", "find_tools", "list_tools"}
        assert channel._exempt_tool_names == {"read_stored_result", "find_tools", "list_tools"}

    def test_a_realtime_configure_refuses_what_the_constructor_refuses(self) -> None:
        """A tool under a name the channel serves itself, or given twice, is
        refused by ``configure(tools=)`` as at construction (RFC §21.1)."""
        channel = RealtimeVoiceChannel(
            "rt", provider=MockRealtimeProvider(), transport=MockRealtimeTransport()
        )

        with pytest.raises(ValueError, match="'find_tools' is a tool channel 'rt' serves"):
            channel.configure(tools=[{"name": "find_tools", "description": "", "parameters": {}}])
        with pytest.raises(ValueError, match="'a' is given twice"):
            channel.configure(tools=[{"name": "a"}, {"name": "a"}])
