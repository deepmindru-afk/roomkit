"""Skill support for RealtimeVoiceChannel.

Handles skill tool definitions, prompt injection, per-session activation
tracking, and tool gating for realtime voice sessions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Container, Iterable
from typing import TYPE_CHECKING, Any

from roomkit.channels._served_tools import dict_tool_name
from roomkit.channels._skill_constants import (
    ACTIVATE_SKILL_SCHEMA,
    READ_REFERENCE_SCHEMA,
    RUN_SCRIPT_SCHEMA,
    SKILL_INFRA_TOOL_NAMES,
    SKILLS_INLINE_PREAMBLE,
    SKILLS_NO_SCRIPTS_NOTE,
    SKILLS_PREAMBLE,
    TOOL_ACTIVATE_SKILL,
    TOOL_READ_REFERENCE,
    TOOL_RUN_SCRIPT,
)
from roomkit.channels._skill_handlers import (
    activation_ack,
    activation_content,
    handle_read_reference,
    handle_run_script,
    missing_skill_error,
    tools_hint,
)
from roomkit.core.exceptions import ToolRefusedError
from roomkit.skills.models import (
    RequiresMatch,
    missing_required_tools,
    missing_tools_error,
    serves_exactly,
)
from roomkit.skills.registry import SkillRegistry
from roomkit.tools.policy import call_gated

if TYPE_CHECKING:
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.models import Skill

logger = logging.getLogger("roomkit.channels.realtime_voice")


SkillDeliveryMode = str
"""``inline_full`` preloads bodies; ``on_demand`` loads only activated skills.

On reconfigurable providers, bodies enter system instructions. Fixed providers
receive the full body in the activation result and must preserve that context.
"""


class RequiredToolsCheck:
    """Whether an activating skill's required tools are among those the
    session declares and its policy admits, read when asked: after the
    activation's hooks ran, before anyone is told (RFC §9.3). A tool only a
    *closed* gate holds does not count (RFC §24.3)."""

    def __init__(
        self,
        skill: Skill | None,
        catalogue: Callable[[], list[dict[str, Any]]],
        closed: Iterable[str] = (),
        match: RequiresMatch = serves_exactly,
    ) -> None:
        self._skill = skill
        self._catalogue = catalogue
        self._closed = list(closed)
        self._match = match
        self.missing: list[str] | None = None
        """The required tools the catalogue lacked, once checked."""

    def held(self) -> bool:
        """Check the catalogue now: whether every required tool is in it."""
        skill = self._skill
        self.missing = (
            missing_required_tools(
                skill.metadata, _names(self._catalogue()), self._closed, match=self._match
            )
            if skill
            else []
        )
        return not self.missing


class RealtimeSkillSupport:
    """Skill delivery and gates scoped to one live conversation.

    Preparing an activation never opens tools. The channel commits it only
    after the provider has accepted its instructions and configuration.
    """

    def __init__(
        self,
        skills: SkillRegistry,
        script_executor: ScriptExecutor | None = None,
        *,
        delivery_mode: SkillDeliveryMode = "on_demand",
        reconfigure_capable: bool = True,
        exempt_tools: Callable[[], Container[str]],
    ) -> None:
        if delivery_mode not in {"inline_full", "on_demand"}:
            raise ValueError(f"Unknown skill delivery mode: {delivery_mode}")
        self._skills = skills
        self._reconfigure_capable = reconfigure_capable
        self._script_executor = script_executor
        self._delivery_mode: SkillDeliveryMode = delivery_mode
        # session_id -> set of activated skill names
        self._activated_skills: dict[str, set[str]] = {}
        # The channel's own tools that only read or unlock, read when asked:
        # skill activation and reference reading, and Tool Search's when the
        # channel has it. Never gated (RFC §21.1).
        self._exempt_tools = exempt_tools
        # session_id -> ordered list of (skill_name, instructions) tuples
        # for skills activated so far in this session. Concatenated into
        # the system_instruction on the next reconfigure_session call.
        # Order matches activation sequence so a chained flow can layer
        # later skills on top of earlier ones.
        self._activated_bodies: dict[str, list[tuple[str, str]]] = {}

    @property
    def delivery_mode(self) -> SkillDeliveryMode:
        return self._delivery_mode

    # -- Tool definitions (as dicts, the format provider.connect expects) --

    def skill_tool_dicts(self) -> list[dict[str, Any]]:
        """Return skill infrastructure tool definitions as plain dicts."""
        tools: list[dict[str, Any]] = [ACTIVATE_SKILL_SCHEMA, READ_REFERENCE_SCHEMA]
        if self._script_executor:
            tools.append(RUN_SCRIPT_SCHEMA)
        return tools

    # -- System prompt injection --

    def inject_skills_prompt(
        self, system_prompt: str | None, *, scripts_allowed: bool = True
    ) -> str:
        """Append skills preamble + available-skills XML to the prompt.

        *scripts_allowed* is whether the session's tool policy admits
        ``run_skill_script``: with no executor or a policy that denies it,
        the model must not be told it can run a skill's scripts (RFC §21.1).

        In ``inline_full`` mode every available skill's full body is
        included verbatim so the model has the binding rules in
        attention from the first token. In ``on_demand`` mode only
        skill metadata is included; bodies arrive later via
        ``provider.reconfigure`` or its tool result after the model calls
        ``activate_skill``.
        """
        if self._delivery_mode == "inline_full":
            preamble = SKILLS_INLINE_PREAMBLE
        else:
            preamble = SKILLS_PREAMBLE
        if not self._script_executor or not scripts_allowed:
            preamble += SKILLS_NO_SCRIPTS_NOTE
        skills_xml = self._skills.to_prompt_xml()
        skill_block = f"\n\n{preamble}\n\n{skills_xml}"

        if self._delivery_mode == "inline_full":
            bodies_block = self._render_all_skill_bodies()
            if bodies_block:
                skill_block += f"\n\n{bodies_block}"

        return (system_prompt or "") + skill_block

    def _render_all_skill_bodies(self) -> str | None:
        """Render every skill's body as a block of binding-rule sections.

        Used by ``inline_full`` mode at session start. Skills without
        instructions are skipped (the metadata XML already advertised
        them; nothing actionable to add).
        """
        sections: list[str] = []
        for meta in self._skills.all_metadata():
            skill = self._skills.get_skill(meta.name)
            body = getattr(skill, "instructions", None) if skill else None
            if not body or not body.strip():
                continue
            sections.append(f"## Skill: {meta.name}\n{body.strip()}")
        if not sections:
            return None
        return "# Loaded skill instructions (binding rules)\n\n" + "\n\n".join(sections)

    # -- Per-session activation tracking --

    def init_session(self, session_id: str) -> None:
        """Initialize activation state for a new session."""
        self._activated_skills[session_id] = set()
        self._activated_bodies[session_id] = []

    def cleanup_session(self, session_id: str) -> None:
        """Remove activation state when a session ends."""
        self._activated_skills.pop(session_id, None)
        self._activated_bodies.pop(session_id, None)

    def activated_skills_prompt(self, session_id: str, pending: Skill | None = None) -> str | None:
        """Return concatenated bodies of skills activated in this session.

        Used by the channel's tool dispatcher: after activate_skill
        runs we call ``provider.reconfigure(system_prompt=base + this)``
        so the skill content lives as binding rules in
        ``system_instruction`` rather than as a giant tool result that
        derails realtime function calling.

        Returns ``None`` when no skills have been activated yet so the
        caller can decide whether a reconfigure is even needed.
        """
        bodies = list(self._activated_bodies.get(session_id) or [])
        if (
            pending
            and self._delivery_mode == "on_demand"
            and pending.name not in self._activated_skills.get(session_id, set())
        ):
            bodies.append((pending.name, pending.instructions))
        if not bodies:
            return None
        sections = [
            f"## Active skill: {name}\n{instructions.strip()}"
            for name, instructions in bodies
            if instructions and instructions.strip()
        ]
        return "\n\n".join(sections) if sections else None

    # -- Tool dispatch --

    def is_skill_tool(self, name: str) -> bool:
        """Return True if *name* is a skill infrastructure tool."""
        return name in SKILL_INFRA_TOOL_NAMES

    async def handle_tool_call(self, name: str, arguments: dict[str, Any], session_id: str) -> str:
        """Dispatch a skill tool call and return the JSON result string."""
        if name == TOOL_ACTIVATE_SKILL:
            return await self._handle_activate_skill(arguments, session_id)
        if name == TOOL_READ_REFERENCE:
            return await self._handle_read_reference(arguments)
        if name == TOOL_RUN_SCRIPT:
            return await self._handle_run_script(arguments)
        return json.dumps({"error": f"Unknown skill tool: {name}"})

    # -- Tool gating --

    def _gated_tool_names(self, session_id: str, pending: Skill | None = None) -> set[str]:
        """Collect tool names gated by skills not yet activated in this session."""
        activated = self._activated_skills.get(session_id, set())
        opened = activated | {pending.name} if pending is not None else activated
        return self._skills.gated_tool_names(opened)

    def is_gated(
        self,
        name: str,
        session_id: str,
        gated: set[str] | None = None,
        *,
        exempt: Container[str] | None = None,
    ) -> bool:
        """Whether *name* is gated by a skill this session has not activated.

        Hiding a tool from the catalogue is not enforcement: a model that saw
        the name before the skill was deactivated — or that read it in a
        transcript — can still call it. Callers ask this at execution time as
        well as at listing time.

        The tools that only read or unlock are never gated when the channel
        serves them itself (RFC §21.1, their ``exempt`` trait): activation and
        reference reading are how a skill gets unlocked, and the Tool Search
        tools are how a gated name is found in the first place. Gating them
        would leave the model told to activate a skill it has no way left to
        name. ``run_skill_script`` acts,
        and is gated like any other tool.

        *gated* lets a caller filtering a whole catalogue compute the gated set
        once instead of once per tool. *exempt* is what escapes on the door
        asking, as its policy reads it (RFC §21.1); by default the channel's
        exempt tools.
        """
        if name in (self._exempt_tools() if exempt is None else exempt):
            return False
        if gated is None:
            gated = self._gated_tool_names(session_id)
        # Under its MCP alias too: the alias runs the gated tool (RFC §21.1).
        return bool(gated) and call_gated(name, gated)

    def get_visible_tools(
        self, all_tools: list[dict[str, Any]], session_id: str, pending: Skill | None = None
    ) -> list[dict[str, Any]]:
        """Filter tool list, removing gated tools but keeping infra tools."""
        gated = self._gated_tool_names(session_id, pending)
        if not gated:
            return all_tools
        # A provider's native tool has no name for a skill to gate: kept.
        return [
            t
            for t in all_tools
            if not (name := dict_tool_name(t)) or not self.is_gated(name, session_id, gated)
        ]

    def newly_visible_after_activation(
        self,
        all_tools: list[dict[str, Any]],
        session_id: str,
        skill_name: str,
    ) -> list[dict[str, Any]] | None:
        """Return updated tool list if activation revealed new tools, else None."""
        meta = self._skills.get_metadata(skill_name)
        if not meta or not meta.gated_tool_names:
            return None
        # Re-filter with the now-activated skill
        return self.get_visible_tools(all_tools, session_id)

    # -- Internal handlers --

    @property
    def uses_tool_result(self) -> bool:
        """Whether the provider must retain dynamically delivered bodies."""
        return self._delivery_mode == "on_demand" and not self._reconfigure_capable

    def commit_activation(self, session_id: str, skill: Skill) -> None:
        """Open gates only after delivery, never resurrecting a closed session."""
        activated = self._activated_skills.get(session_id)
        if activated is not None and skill.name not in activated:
            activated.add(skill.name)
            if self._delivery_mode == "on_demand":
                self._activated_bodies[session_id].append((skill.name, skill.instructions))

    def unknown_skill_hint(
        self, result: str, skill_name: str, reachable: Iterable[str], *, call_tool: bool = False
    ) -> tuple[str, list[str]]:
        """*result* of an activation that found no skill *skill_name*, with the
        tools among *reachable* its name matches, hinted, and those tools."""
        return tools_hint(result, skill_name, self._skills, reachable, call_tool=call_tool)

    @property
    def requires_match(self) -> RequiresMatch:
        """How the registry reads a skill's ``requires`` names (RFC §24.3)."""
        return self._skills.requires_match

    def is_closed_for_good(self, name: str) -> bool:
        """Whether only skills marked unavailable gate *name*: no activation
        can open it (RFC §24.2)."""
        return call_gated(name, self._skills.unopenable_tool_names())

    def _serving_tools(
        self, skill: Skill, catalogue: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """The schemas of the tools that serve *skill*'s ``requires``, as the
        registry's ``requires_match`` reads them, each once (RFC §24.3)."""
        match = self._skills.requires_match
        serving: dict[str, dict[str, Any]] = {}
        for required in skill.metadata.required_tool_names:
            for name, tool in catalogue.items():
                if match(required, (name,)):
                    serving.setdefault(name, tool)
        return list(serving.values())

    def closed_for(self, skill: Skill, session_id: str) -> set[str]:
        """The closed gates an activation of *skill* meets: those no skill of
        the session's, *skill* included, can open (RFC §24.2)."""
        opened = self._activated_skills.get(session_id, set()) | {skill.name}
        return self._skills.closed_tool_names(opened)

    async def prepare_activation(
        self, arguments: dict[str, Any], session_id: str, tools: list[dict[str, Any]]
    ) -> tuple[str, Skill | None]:
        """Build an immutable delivery candidate from the authorized catalogue."""
        skill_name = arguments.get("name", "")
        skill = await asyncio.to_thread(self._skills.get_skill, skill_name)
        if skill is None:
            return json.dumps(
                {
                    "error": missing_skill_error(self._skills, skill_name),
                    "available_skills": self._skills.skill_names,
                }
            ), None
        catalogue = {name: tool for tool in tools if (name := dict_tool_name(tool))}
        missing = missing_required_tools(
            skill.metadata,
            _names(tools),
            self.closed_for(skill, session_id),
            match=self._skills.requires_match,
        )
        if missing:
            # A refusal, as the activation itself refuses it (RMK-395).
            raise ToolRefusedError(missing_tools_error(missing))

        if self.uses_tool_result:
            result = await asyncio.to_thread(activation_content, skill)
            payload = json.loads(result)
            payload["ok"] = True
            payload["_note"] = (
                "Follow these complete skill instructions for this session. "
                "Use the required tool schemas below; tool names and actions are distinct."
            )
            payload["required_tools"] = self._serving_tools(skill, catalogue)
            if skill_name in self._activated_skills.get(session_id, set()):
                payload["already_active"] = True
            return json.dumps(payload), skill

        note = (
            "The skill instructions are already loaded in your system rules. Follow them."
            if self._delivery_mode == "inline_full"
            else "Loading the skill instructions into your system rules before continuing."
        )
        result = await asyncio.to_thread(
            activation_ack,
            skill,
            note,
            already_active=skill_name in self._activated_skills.get(session_id, set()),
        )
        return result, skill

    async def _handle_activate_skill(self, arguments: dict[str, Any], session_id: str) -> str:
        """Prepare a result; the channel owns delivery and activation commit."""
        try:
            result, _ = await self.prepare_activation(arguments, session_id, [])
        except ToolRefusedError as refusal:
            return refusal.message
        return result

    async def _handle_read_reference(self, arguments: dict[str, Any]) -> str:
        """Read a reference file from a skill."""
        return await handle_read_reference(arguments, self._skills)

    async def _handle_run_script(self, arguments: dict[str, Any]) -> str:
        """Execute a script via the configured ScriptExecutor."""
        return await handle_run_script(arguments, self._skills, self._script_executor)


def _names(tools: list[dict[str, Any]]) -> list[str]:
    return [name for tool in tools if (name := dict_tool_name(tool))]
