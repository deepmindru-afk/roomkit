"""A room's declaration is kept from one turn to the next (RFC §6.4, RMK-345).

Where the provider holds a tool declared but unseen, a tool an earlier turn
opened (a reveal, an unlock) stays held rather than joining the tools shown,
so the next turn's declaration, the head of the cached prefix, is the same.
The turn reopens it with an exchange before its input: a call of the
discovery tool the turn declares, and a result that references it.
"""

from __future__ import annotations

from roomkit.channels._tool_reopen import room_declaration
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import RetryPolicy
from roomkit.providers.ai.base import AIContext, AIToolCallPart, AIToolResultPart
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills import SkillRegistry
from tests.test_deferred_tools import (
    _CATALOGUE,
    _DONE,
    _FIXTURES,
    FailingHolder,
    HoldingProvider,
    _declaration,
    _round,
    _searching,
    _served,
    _tool,
    _turn,
)

_REVEAL = _round("c0", "find_tools", {"query": "track shipment", "max_results": 1})


def _reopening(context: AIContext) -> tuple[AIToolCallPart, AIToolResultPart]:
    """The exchange before the turn's input: its call and its result. (The
    mock keeps the round's context, whose messages the loop goes on to grow.)"""
    turn_input = next(i for i, m in enumerate(context.messages) if m.content == "go")
    call_message, result_message = context.messages[turn_input - 2 : turn_input]
    [call] = call_message.content
    [result] = result_message.content
    assert isinstance(call, AIToolCallPart) and isinstance(result, AIToolResultPart)
    return call, result


async def test_a_tool_revealed_in_one_turn_is_reopened_in_the_next(streaming: bool) -> None:
    provider = HoldingProvider(
        ai_responses=[_REVEAL, _DONE, _round("c1", "track_shipment"), _DONE],
        streaming=streaming,
    )
    ch = _searching(provider)

    await _turn(ch, _CATALOGUE)
    run = await _turn(ch, _CATALOGUE)

    first_turn, next_turn = provider.calls[0], provider.calls[2]
    assert _declaration(next_turn) == _declaration(first_turn)
    assert ("track_shipment", True) in _declaration(next_turn)
    call, result = _reopening(next_turn)
    assert (call.name, result.name, result.references) == (
        "find_tools",
        "find_tools",
        ["track_shipment"],
    )
    # Callable, and the exchange is context: it ran nothing and recorded nothing.
    assert [c.name for c in run.calls] == ["track_shipment"]
    assert not run.calls[0].failed


async def test_a_tool_a_skill_unlocked_is_reopened_with_the_skill(streaming: bool) -> None:
    registry = SkillRegistry()
    registry.discover(_FIXTURES)
    provider = HoldingProvider(
        ai_responses=[
            _round("c0", "activate_skill", {"name": "quote-policy"}),
            _DONE,
            _round("c1", "inventory"),
            _DONE,
        ],
        streaming=streaming,
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=_served, skills=registry)
    tools = [_tool("lookup"), _tool("inventory")]

    await _turn(ch, tools)
    run = await _turn(ch, tools)

    assert _declaration(provider.calls[2]) == _declaration(provider.calls[0])
    call, result = _reopening(provider.calls[2])
    assert (call.name, call.arguments, result.references) == (
        "activate_skill",
        {"name": "quote-policy"},
        ["inventory"],
    )
    assert [c.name for c in run.calls] == ["inventory"]
    assert not run.calls[0].failed


async def test_a_fallback_that_cannot_hold_receives_the_reopened_tool(streaming: bool) -> None:
    fallback = MockAIProvider(ai_responses=[_DONE], streaming=streaming)
    ch = _searching(
        HoldingProvider(ai_responses=[_REVEAL, _DONE], streaming=streaming),
        fallback_provider=fallback,
        retry_policy=RetryPolicy(max_retries=0),
    )
    await _turn(ch, _CATALOGUE)
    ch._provider = FailingHolder(streaming=streaming)

    await _turn(ch, _CATALOGUE)

    declared = _declaration(fallback.calls[0])
    assert ("track_shipment", False) in declared
    assert all(not held for _, held in declared)


async def test_a_provider_that_cannot_hold_shows_the_tool_it_opened(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=[_REVEAL, _DONE, _DONE], streaming=streaming)
    ch = _searching(provider)

    await _turn(ch, _CATALOGUE)
    await _turn(ch, _CATALOGUE)

    next_turn = provider.calls[2]
    assert "track_shipment" in {t.name for t in next_turn.tools}
    assert not any(isinstance(p, AIToolCallPart) for m in next_turn.messages for p in m.content)


def test_the_room_declaration_keeps_what_its_first_turn_showed() -> None:
    kept = frozenset({"lookup", "find_tools"})
    shown = {"lookup", "find_tools", "track_shipment", "plan_tasks"}

    assert room_declaration(None, shown, set()) == frozenset(shown)
    # A tool the channel never holds joins; one opened since stays held.
    assert room_declaration(kept, shown, {"plan_tasks", "find_tools"}) == frozenset(
        {"lookup", "find_tools", "plan_tasks"}
    )
    # One the turn no longer shows (the policy denies it now) leaves.
    assert room_declaration(kept, {"find_tools"}, set()) == frozenset({"find_tools"})


async def test_a_pinned_tool_a_skill_gates_is_reopened_not_shown(streaming: bool) -> None:
    """A host's pinned tool is held while a skill gates it; once a turn opened
    it, the next reopens it like any other rather than showing it."""
    registry = SkillRegistry()
    registry.discover(_FIXTURES)
    provider = HoldingProvider(
        ai_responses=[
            _round("c0", "activate_skill", {"name": "quote-policy"}),
            _DONE,
            _round("c1", "inventory"),
            _DONE,
        ],
        streaming=streaming,
    )
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=_served,
        tool_search=True,
        tool_search_pinned={"lookup", "inventory"},
        skills=registry,
    )
    tools = [*_CATALOGUE, _tool("inventory")]

    await _turn(ch, tools)
    run = await _turn(ch, tools)

    assert _declaration(provider.calls[2]) == _declaration(provider.calls[0])
    _call, result = _reopening(provider.calls[2])
    assert result.references == ["inventory"]
    assert [c.name for c in run.calls] == ["inventory"] and not run.calls[0].failed
