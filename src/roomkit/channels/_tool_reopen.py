"""A turn's declaration kept from the room's, and the tools it reopens (RFC §6.4).

On a provider that holds a tool declared but unseen, a prompt cache reads the
request as a prefix, tools first: a turn that shows one more tool than the
last rewrites the whole history after it. So the tools a room's turns show
stay the ones its first turn showed, and a tool the turn would show beyond
them (one an earlier turn opened, one added since) stays held. The reference
that made it callable rode a tool result the next turn's history does not
replay, so the turn reopens it with an exchange of its own: a call of the
discovery tool the turn declares, and a result that references it. The
exchange is context, never an event of the room.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.channels._skill_constants import ALREADY_ACTIVE_NOTE, TOOL_ACTIVATE_SKILL
from roomkit.channels._skill_handlers import activation_ack
from roomkit.channels._tool_search import render_find_payload
from roomkit.channels._tool_search_constants import TOOL_FIND_TOOLS
from roomkit.providers.ai.base import AIMessage, AITool, AIToolCallPart
from roomkit.tools._outcome import OutcomeKind, ToolOutcome

if TYPE_CHECKING:
    from roomkit.skills.registry import SkillRegistry

# Marks the exchange's messages, which a provider that cannot hold a tool
# does not receive (``declared_for``).
REOPENING = "reopening"
# What the reopening find_tools call asks, as the model reads it.
_REOPEN_QUERY = "tools this conversation can call"


def room_declaration(
    kept: frozenset[str] | None, shown: set[str], never_held: set[str]
) -> frozenset[str]:
    """The tools this turn shows: the room's *kept* declaration still shown,
    with every tool the channel never holds; all of *shown* for a room whose
    declaration is not known (a new room, a restarted process)."""
    if kept is None:
        return frozenset(shown)
    return frozenset((kept & shown) | (shown & never_held))


def reopening_exchanges(
    opened: list[AITool],
    first: frozenset[str],
    active: set[str],
    skills: SkillRegistry | None,
    call_id: str,
) -> tuple[list[AIMessage], list[AITool]]:
    """The exchanges that reopen *opened*, and the tools none reaches.

    One ``find_tools`` call where the turn declares it; otherwise one
    ``activate_skill`` call for each *active* skill that gates a tool of
    *opened*, answered as a call to an active skill is.
    """
    if TOOL_FIND_TOOLS in first:
        matches = [{"name": t.name, "description": t.description} for t in opened]
        result = render_find_payload(matches)
        return _exchange(call_id, TOOL_FIND_TOOLS, {"query": _REOPEN_QUERY}, result, opened), []
    if TOOL_ACTIVATE_SKILL not in first or skills is None:
        return [], opened
    exchanges: list[AIMessage] = []
    left = list(opened)
    for name in sorted(active):
        skill = skills.get_skill(name)
        gated = [t for t in left if skill is not None and skill.metadata.gates(t.name)]
        if skill is None or not gated:
            continue
        ack = activation_ack(skill, ALREADY_ACTIVE_NOTE, already_active=True)
        call = f"{call_id}_{len(exchanges) // 2}"
        exchanges += _exchange(call, TOOL_ACTIVATE_SKILL, {"name": name}, ack, gated)
        left = [t for t in left if t not in gated]
    return exchanges, left


def insert_before(
    messages: list[AIMessage], turn_input: AIMessage | None, exchanges: list[AIMessage]
) -> bool:
    """Insert *exchanges* right before *turn_input*; ``False`` when the turn
    has no input among *messages* to place them before."""
    index = next((i for i, m in enumerate(messages) if m is turn_input), None)
    if index is None:
        return False
    messages[index:index] = exchanges
    return True


def _exchange(
    call_id: str, tool: str, arguments: dict[str, str], result: str, opened: list[AITool]
) -> list[AIMessage]:
    """A call of *tool* and its *result*, which references *opened*."""
    marker = {REOPENING: True}
    call = AIToolCallPart(id=call_id, name=tool, arguments=arguments)
    answer = ToolOutcome(OutcomeKind.SERVED, result).as_part(
        call_id, tool, references=[t.name for t in opened]
    )
    return [
        AIMessage(role="assistant", content=[call], metadata=marker),
        AIMessage(role="tool", content=[answer], metadata=marker),
    ]
