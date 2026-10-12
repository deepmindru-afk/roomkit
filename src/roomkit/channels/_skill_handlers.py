"""Shared skill tool handlers used by both AIChannel and RealtimeVoiceChannel."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from roomkit._text import fence
from roomkit.channels._skill_constants import (
    ALREADY_ACTIVE_NOTE,
    SERVED_ALREADY_ACTIVE_NOTE,
    SERVED_SKILL_NOTE,
)
from roomkit.core.exceptions import ToolRefusedError
from roomkit.skills.errors import SkillError, SkillPathError
from roomkit.skills.mcp.skill import MCPSkill

if TYPE_CHECKING:
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.models import Skill
    from roomkit.skills.registry import SkillRegistry

logger = logging.getLogger("roomkit.channels.skills")


def missing_skill_error(skills: SkillRegistry, skill_name: str) -> str:
    """Error line for a skill ``get_skill()`` could not return.

    A skill marked unavailable gets its reason (e.g. a required tool is
    not granted in this context) instead of a misleading "not found".
    """
    reason = skills.get_unavailable_reason(skill_name)
    if reason is not None:
        return f"Skill {skill_name!r} is unavailable in this context: {reason}"
    return f"Skill {skill_name!r} not found"


def tools_hint(
    result: str,
    skill_name: str,
    skills: SkillRegistry,
    reachable: Iterable[str],
    *,
    call_tool: bool = False,
) -> tuple[str, list[str]]:
    """*result* of an activation that found no skill *skill_name*, with a hint
    naming the tools the turn may call that match it, and those tools, which
    the caller reveals as ``find_tools`` reveals its matches.

    Small models routinely confuse skills with tools ("activate the Spotify
    skill" when ``spotify_play`` is a tool): the dead end becomes the right
    outcome. A known skill that is unavailable gets no hint: its error
    already says why, and a "this is not a skill" hint would contradict it.
    With *call_tool* (a provider whose declarations are fixed), the tools
    are reached through ``list_tools`` and ``call_tool``, as ``find_tools``
    says of its matches there.
    """
    wanted = skill_name.lower()
    if not wanted or skills.get_unavailable_reason(skill_name) is not None:
        return result, []
    matching = sorted(name for name in reachable if wanted in name.lower())
    if not matching:
        return result, []
    named = ", ".join(matching[:8])
    data = json.loads(result)
    data["tools_hint"] = (
        f"{skill_name!r} is not a skill, but these TOOLS match — read one's "
        f"complete schema with list_tools(name=...), then call it with call_tool: {named}."
        if call_tool
        else f"{skill_name!r} is not a skill, but these TOOLS match and are now in your "
        f"tool list — call one directly instead: {named}."
    )
    return json.dumps(data), matching


def already_active_note(skill: Skill) -> str:
    """What an acknowledgement says of a skill already active: where its
    content is in force (RFC §24.4), a served skill's apart (RFC §24.7)."""
    return SERVED_ALREADY_ACTIVE_NOTE if isinstance(skill, MCPSkill) else ALREADY_ACTIVE_NOTE


def _origin(skill: Skill) -> dict[str, Any]:
    """The origin a served skill's content names as it reaches the model
    (RFC §24.7); nothing for a host's own skill."""
    if not isinstance(skill, MCPSkill):
        return {}
    return {"origin": skill.origin, "uri": skill.held.uri, "origin_note": SERVED_SKILL_NOTE}


def activation_ack(skill: Skill, note: str, *, already_active: bool = False) -> str:
    """Acknowledge an activation without re-sending the skill's body.

    The body reaches the model through the system prompt instead — folded into
    ``system_instruction`` on realtime channels, into the turn's system prompt on
    the text channel. What the tool still owes the model is the activation itself
    plus the skill's file inventory, which is what this returns.
    """
    payload: dict[str, Any] = {"ok": True, "name": skill.name, "_note": note, **_origin(skill)}
    if already_active:
        payload["already_active"] = True
    scripts = skill.list_scripts()
    if scripts:
        payload["scripts"] = scripts
    refs = skill.list_references()
    if refs:
        payload["references"] = refs
    return json.dumps(payload)


async def handle_activate_skill(
    arguments: dict[str, Any],
    skills: SkillRegistry,
) -> tuple[str, str, Skill | None]:
    """Load a skill and return its full instructions.

    Returns:
        ``(json_result, skill_name, skill)``: the skill this activation
        loaded, ``None`` when it loaded none. A caller tracks the activation
        from it, never from a second lookup, which could find a skill the
        load refused (one loaded before, now unavailable).
    """
    skill_name = arguments.get("name", "")
    try:
        skill = await skills.load_skill(skill_name)
    except SkillError as exc:
        # A served skill not found on its server, or whose content does not
        # match its entry: the message names the skill only (RFC §24.7).
        error = {"error": str(exc), "available_skills": skills.skill_names}
        return json.dumps(error), skill_name, None
    if skill is None:
        error_json = json.dumps(
            {
                "error": missing_skill_error(skills, skill_name),
                "available_skills": skills.skill_names,
            }
        )
        return error_json, skill_name, None

    return await asyncio.to_thread(activation_content, skill), skill_name, skill


def activation_content(skill: Skill) -> str:
    """Serialize full instructions and the file inventory without truncation."""
    # A served skill's content goes out set apart as its server's, in a block
    # it cannot close, wherever it travels (RFC §6.4, §24.7).
    instructions = (
        fence("skill", skill.instructions) if isinstance(skill, MCPSkill) else skill.instructions
    )
    payload: dict[str, Any] = {
        "name": skill.name,
        "description": skill.description,
        "instructions": instructions,
        **_origin(skill),
    }
    scripts = skill.list_scripts()
    if scripts:
        payload["scripts"] = scripts
    refs = skill.list_references()
    if refs:
        payload["references"] = refs
    return json.dumps(payload)


async def handle_read_reference(
    arguments: dict[str, Any],
    skills: SkillRegistry,
) -> str:
    """Read a reference file from a skill."""
    skill_name = arguments.get("skill_name", "")
    filename = arguments.get("filename", "")
    try:
        content = await skills.read_file(skill_name, filename)
    except KeyError:
        # A skill the registry does not offer: the call is refused, as a
        # call to a tool it cannot reach is (RFC §9.3). A served skill's
        # files are read once it is active (RFC §24.7).
        error = (
            f"Skill {skill_name!r} is not active: activate it before reading its files"
            if skills.is_served(skill_name) and skill_name in skills.skill_names
            else missing_skill_error(skills, skill_name)
        )
        raise ToolRefusedError(json.dumps({"error": error})) from None
    except (ValueError, FileNotFoundError) as exc:
        return json.dumps({"error": str(exc)})
    payload: dict[str, Any] = {"filename": filename, "content": content}
    origin = skills.origin(skill_name)
    if origin is not None:
        payload["origin"] = origin.label
    return json.dumps(payload)


async def handle_run_script(
    arguments: dict[str, Any],
    skills: SkillRegistry,
    script_executor: ScriptExecutor | None,
) -> str:
    """Execute a script via the configured ScriptExecutor."""
    skill_name = arguments.get("skill_name", "")
    script_name = arguments.get("script_name", "")
    script_args = arguments.get("arguments")
    if not script_executor:
        return json.dumps({"error": "Script execution is not available"})

    skill = skills.get_skill(skill_name)
    if skill is None:
        # Refused, as a reference read from a skill the registry does not offer is.
        raise ToolRefusedError(json.dumps({"error": missing_skill_error(skills, skill_name)}))

    # Which file runs is the framework's call; how it runs is the executor's.
    # Resolving here means a name that escapes the skill never reaches an
    # integrator's executor, rather than relying on each one to re-check it.
    try:
        skill.resolve_script(script_name)
    except (SkillPathError, FileNotFoundError) as exc:
        return json.dumps({"error": str(exc)})

    # A failure propagates: the channel reads it as any raised call, with its
    # marker, the class for the model and the message for the observers
    # (RFC §9.3).
    result = await script_executor.execute(skill, script_name, arguments=script_args)
    return result.model_dump_json()
