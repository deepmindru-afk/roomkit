"""AIChannel mixin for tool policy enforcement and skill gating."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from roomkit.channels._skill_constants import (
    SKILL_INFRA_TOOL_NAMES,
    TOOL_ACTIVATE_SKILL,
    TOOL_READ_REFERENCE,
)
from roomkit.channels._tool_search_constants import (
    TOOL_FIND_TOOLS,
    TOOL_LIST_TOOLS,
    TOOL_SEARCH_INFRA_TOOL_NAMES,
)
from roomkit.models.tool_call import DeclaredTool, ToolDeclarationOrigin
from roomkit.providers.ai.base import AITool
from roomkit.tools.policy import ToolPolicy, matches_any_pattern

if TYPE_CHECKING:
    from collections.abc import Iterable

    from roomkit.channels._skill_activation import SkillActivationMemory
    from roomkit.channels.ai import _ToolLoopContext
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.skills.registry import SkillRegistry

logger = logging.getLogger("roomkit.channels.ai")

# RFC §21.1: the tools a channel provides that only read or unlock and never
# act. They escape the tool policy and skill gating, by exact name, wherever
# either is applied: the declared list, the execution guards, the Tool Search
# catalogue. Every other tool the channel injects (sandbox commands,
# run_skill_script, plan_tasks) is governed like a host tool.
POLICY_EXEMPT_TOOL_NAMES: frozenset[str] = frozenset(
    {
        TOOL_ACTIVATE_SKILL,
        TOOL_READ_REFERENCE,
        "read_stored_result",
        TOOL_FIND_TOOLS,
        TOOL_LIST_TOOLS,
    }
)


def policy_admits(policy: ToolPolicy | None, name: str) -> bool:
    """Whether a channel's role-resolved *policy* admits *name* (RFC §21.1).

    The one reading of the policy for every channel that carries one, AI or
    realtime, and for a conference: the exempt names pass, anything else is
    the policy's to allow.
    """
    return name in POLICY_EXEMPT_TOOL_NAMES or policy is None or policy.is_allowed(name)


def policy_refusal(name: str) -> str:
    """What the model reads of a call the tool policy refused."""
    return f"Tool '{name}' is not permitted by the agent's tool policy."


@runtime_checkable
class ToolPolicyHost(Protocol):
    """Contract: capabilities a host class must provide for AIToolPolicyMixin.

    Attributes provided by the host's ``__init__``:
        _tool_policy: Global tool access policy (may contain per-role overrides).
        _skills: Skill registry for gated tool resolution.
        _skill_activation: Per-room record of the skills active in a conversation.

    Methods provided by AISteeringMixin (or equivalent):
        _get_loop_ctx: Return the current tool-loop context (activated skills,
            participant role, steering queue).

    Methods provided by AIChannel:
        _orchestration_tool_names: The tools orchestration injected for a
            room, which Tool Search never defers.
    """

    _tool_policy: ToolPolicy | None
    _skills: SkillRegistry | None
    _skill_activation: SkillActivationMemory
    _tool_search_pinned: set[str]

    def _get_loop_ctx(self) -> _ToolLoopContext: ...
    def _orchestration_tool_names(self, room_id: str | None) -> set[str]: ...


class AIToolPolicyMixin:
    """Resolves participant roles and enforces tool policy / skill gating.

    Host contract: :class:`ToolPolicyHost`.
    """

    _tool_policy: ToolPolicy | None
    _skills: SkillRegistry | None
    _skill_activation: SkillActivationMemory
    _tool_search_pinned: set[str]
    _get_loop_ctx: Callable[[], _ToolLoopContext]
    _orchestration_tool_names: Callable[[str | None], set[str]]

    def _resolve_participant_role(self, event: RoomEvent, context: RoomContext) -> str | None:
        """Look up the participant role for the event source."""
        pid = event.source.participant_id
        if not pid:
            return None
        for p in context.participants:
            if p.id == pid:
                return p.role
        return None

    @property
    def _effective_tool_policy(self) -> ToolPolicy | None:
        """Return the tool policy resolved for the current participant role."""
        if self._tool_policy is None:
            return None
        return self._tool_policy.resolve(self._get_loop_ctx().current_participant_role)

    # Channel-managed tool names: dispatched by the channel itself and never
    # deferred by Tool Search. Not an exemption from the policy — that is
    # POLICY_EXEMPT_TOOL_NAMES, a narrower set.
    _SKILL_INFRA_TOOLS: frozenset[str] = SKILL_INFRA_TOOL_NAMES | frozenset(
        {"read_stored_result", "plan_tasks"}
    )
    _NEVER_DEFERRED: frozenset[str] = _SKILL_INFRA_TOOLS | TOOL_SEARCH_INFRA_TOOL_NAMES

    @property
    def _gated_tool_names(self) -> set[str]:
        """Collect tool names gated by skills that have NOT been activated yet.

        An activation counts for the whole conversation, not just the turn that
        made it: the loop context is only this turn's view, so it is unioned
        with the room's activation record. Without that, a skill activated in
        turn N would see its gated tools disappear again in turn N+1 — while
        its instructions, now carried by the system prompt, still tell the
        model to use them.
        """
        if not self._skills:
            return set()
        loop_ctx = self._get_loop_ctx()
        activated = loop_ctx.activated_skills | self._skill_activation.active_names(
            loop_ctx.room_id
        )
        gated: set[str] = set()
        for meta in self._skills.all_metadata():
            if meta.name in activated:
                continue
            gated.update(meta.gated_tool_names)
        return gated

    def _record_declared_tools(
        self, loop_ctx: _ToolLoopContext, tools: list[AITool] | None
    ) -> None:
        """Record the toolset one generation round hands the provider.

        Called with the ``AIContext.tools`` of every provider call of the turn,
        by the loop that makes the call. Not from ``_apply_tool_filters``: its
        output is not the round's declaration (a ``BEFORE_AI_GENERATION`` hook
        may edit the tools, the eviction tool is injected per round beside it,
        the force-stop ripcord strips them) and it also serves as a single-tool
        probe. A name is recorded once per turn, on its first round, with the
        reason Tool Search let it through as it stood then. The turn's
        ``AIResponseEvent.declared_tools`` reports the union.
        """
        if not tools:
            return
        declared = loop_ctx.declared_tools
        for tool in tools:
            if tool.name not in declared:
                origin = self._declaration_origin(tool.name, loop_ctx)
                declared[tool.name] = DeclaredTool.from_tool(tool, origin)

    def _declaration_origin(self, name: str, loop_ctx: _ToolLoopContext) -> ToolDeclarationOrigin:
        """Why a tool is in this round's declaration (``ToolDeclarationOrigin``).

        The keep-set of ``_apply_tool_filters`` is ``pinned | revealed |
        sticky`` plus the tools the collapse never touches; this names which
        term admitted the tool, pinned before sticky before revealed: the
        earliest reason it was visible. Outside Tool Search, and for a tool
        the collapse never touches (an infrastructure tool, one orchestration
        injected, one a hook added), there is no such reason (RFC §21.1).
        """
        if not loop_ctx.tool_search_active:
            return "always"
        if name in self._tool_search_pinned:
            return "pinned"
        if name in self._orchestration_tool_names(loop_ctx.room_id) | loop_ctx.hook_pinned:
            return "always"
        if name in loop_ctx.sticky_tools:
            return "sticky"
        if name in loop_ctx.revealed_tools:
            return "revealed"
        return "always"

    def _policy_allows(self, name: str) -> bool:
        """Whether the turn's role-resolved policy admits *name* (RFC §21.1).

        Skill gating aside: what the prompt may describe as available, a gated
        tool included, since activating its skill opens it.
        """
        return policy_admits(self._effective_tool_policy, name)

    def _gate_refusal(self, name: str) -> dict[str, str] | None:
        """Why the policy or skill gating refuses a call to *name*, or ``None``.

        The execution guard's reading of the listing filter's rule (RFC
        §21.1): the same exempt names, the same role-resolved policy, the same
        glob-aware gating (RFC §24.2).
        """
        if name in POLICY_EXEMPT_TOOL_NAMES:
            return None
        if not policy_admits(self._effective_tool_policy, name):
            logger.warning("Tool %s blocked by policy", name)
            return {"error": policy_refusal(name)}
        if matches_any_pattern(name, self._gated_tool_names):
            logger.warning("Tool %s blocked by skill gating", name)
            return {
                "error": (
                    f"Tool '{name}' is gated by a skill. "
                    "Activate the skill first using activate_skill."
                )
            }
        return None

    def _reachable_tools(self, tools: Iterable[AITool]) -> list[AITool]:
        """The tools the policy and skill gating let this turn reach (RFC §21.1).

        What ``find_tools`` and ``list_tools`` search and hint from: a name the
        model could never call is a false promise, and naming it discloses
        what the policy hides. Tool Search's reveal window does not apply.
        """
        policy = self._effective_tool_policy
        gated = self._gated_tool_names
        return [tool for tool in tools if self._is_reachable(tool.name, policy, gated)]

    @staticmethod
    def _is_reachable(name: str, policy: ToolPolicy | None, gated: set[str]) -> bool:
        """Whether the role-resolved *policy* and skill gating admit *name*."""
        if name in POLICY_EXEMPT_TOOL_NAMES:
            return True
        if not policy_admits(policy, name):
            return False
        # ``gated`` holds ToolPolicy globs, not names (RFC §24.2): an
        # exact-membership test would let ``search_*`` gate nothing at all.
        return not matches_any_pattern(name, gated)

    def _apply_tool_filters(self, tools: list[AITool]) -> list[AITool]:
        """Apply tool policy, skill gating, and Tool Search to a list of tools.

        The policy-exempt tools (``POLICY_EXEMPT_TOOL_NAMES``: skill activation
        and reference reading, the eviction re-read, Tool Search's discovery
        tools) always pass: they ARE the skill/discovery mechanism and must
        stay reachable while the discretionary catalogue is hidden. Every
        other tool, the ones the channel injects included, passes the
        role-aware policy and skill gating first (RFC §21.1).

        When Tool Search is active for the turn (``loop_ctx.tool_search_active``,
        set in ``_build_context``), the discretionary catalogue is collapsed to
        the pinned set plus the tools already revealed by ``find_tools`` this
        loop. The re-filter runs every round, so a tool revealed in round N
        becomes visible in round N+1 — the same mechanism as skill gating.
        Channel-managed tools (``run_skill_script``, ``plan_tasks``) are never
        deferred, nor are the tools orchestration injected for this room (RFC
        §21.1): the agent is told to call them. Sandbox tools are: a sandbox
        can expose ~10 tools, which on a small model would crowd the context
        window, so they wait behind ``find_tools`` like any other
        discretionary tool unless the host pins them via ``tool_search_pinned``.
        """
        gated = self._gated_tool_names
        policy = self._effective_tool_policy
        loop_ctx = self._get_loop_ctx()
        # ``None`` = Tool Search inactive (no collapse). Otherwise the set of
        # discretionary tool names that stay visible this round.
        keep: set[str] | None = None
        if loop_ctx.tool_search_active:
            # pinned (config) + revealed (find_tools this loop) + sticky (tools
            # already used this conversation, re-exposed so they stay callable)
            # + the channel's own, orchestration's and the generation hook's
            # additions, never deferred.
            keep = (
                self._tool_search_pinned
                | loop_ctx.revealed_tools
                | loop_ctx.sticky_tools
                | self._NEVER_DEFERRED
                | self._orchestration_tool_names(loop_ctx.room_id)
                | loop_ctx.hook_pinned
            )
        result: list[AITool] = []
        for tool in tools:
            name = tool.name
            if not self._is_reachable(name, policy, gated):
                continue
            if keep is not None and name not in keep:
                continue
            result.append(tool)
        return result
