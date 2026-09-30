"""A turn's declaration kept from the room's, and the tools it reopens (RFC §6.4).

On a provider that holds a tool declared but unseen, a prompt cache reads the
request as a prefix, tools first: a turn that shows one more tool than the
last rewrites the whole history after it. So the tools a room's turns show
stay the ones its first turn showed, and a tool an earlier turn opened (a
reveal, a use, a skill's unlock) stays held. Its reference, which made it
callable, rode a tool result the next turn's history does not replay, so the
turn reopens it with an exchange of its own: a call of the discovery tool the
turn declares, and a result that references it. The exchange is context,
never an event of the room.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from roomkit.channels._skill_constants import TOOL_ACTIVATE_SKILL
from roomkit.channels._tool_search import render_find_payload
from roomkit.channels._tool_search_constants import TOOL_FIND_TOOLS
from roomkit.providers.ai.base import AIMessage, AITool, AIToolCallPart, AIToolResultPart

# What the reopening find_tools call asks, as the model reads it.
_REOPEN_QUERY = "tools opened earlier in this conversation"


def room_declaration(
    kept: frozenset[str] | None, shown: set[str], never_held: set[str]
) -> frozenset[str]:
    """The tools this turn shows: the room's *kept* declaration still shown,
    with every tool the channel never holds; all of *shown* for a room whose
    declaration is not known (a new room, a restarted process)."""
    if kept is None:
        return frozenset(shown)
    return frozenset((kept & shown) | (shown & never_held))


def reopen(
    messages: list[AIMessage],
    turn_input: AIMessage | None,
    opened: Sequence[AITool],
    first: frozenset[str],
    active_skills: set[str],
    call_id: str,
) -> bool:
    """Insert, before *turn_input*, the exchange that makes *opened* callable.

    The call is ``find_tools`` where the turn declares it, else
    ``activate_skill`` for a skill already active. ``False`` when the turn
    declares neither or has no input to place it before: the caller then
    declares *opened* plainly.
    """
    index = next((i for i, m in enumerate(messages) if m is turn_input), None)
    names = [t.name for t in opened]
    if index is None:
        return False
    if TOOL_FIND_TOOLS in first:
        arguments: dict[str, str] = {"query": _REOPEN_QUERY}
        matches = [{"name": t.name, "description": t.description} for t in opened]
        result = render_find_payload(matches)
        tool = TOOL_FIND_TOOLS
    elif TOOL_ACTIVATE_SKILL in first and active_skills:
        skill = sorted(active_skills)[0]
        arguments = {"name": skill}
        result = json.dumps({"skill": skill, "already_active": True, "tools": names})
        tool = TOOL_ACTIVATE_SKILL
    else:
        return False
    messages[index:index] = [
        AIMessage(
            role="assistant",
            content=[AIToolCallPart(id=call_id, name=tool, arguments=arguments)],
        ),
        AIMessage(
            role="tool",
            content=[
                AIToolResultPart(tool_call_id=call_id, name=tool, result=result, references=names)
            ],
        ),
    ]
    return True
