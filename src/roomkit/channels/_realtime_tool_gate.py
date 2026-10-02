"""The pre-execution gate of a RealtimeVoiceChannel's tool calls (RFC §12.4).

What a session declares (its catalogue, what orchestration adds for its room,
the channel's own tools, each name once), what its tool policy admits for its
participant's role, and the gate a call passes before anything serves it, in
RFC §12.4's order: the declared tool, its schema (a flattened hub call folded
back first), the policy, skill gating, BEFORE_TOOL_USE, whose arguments are
validated again.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Container
from typing import TYPE_CHECKING, Any

from roomkit.channels._ai_policy import policy_admits, policy_refusal
from roomkit.channels._served_tools import CollisionLog, declared_once, dict_tool_name
from roomkit.channels._tool_registry import ChannelRegistry, ToolSource, tool_dict
from roomkit.models.enums import ChannelType
from roomkit.models.tool_call import ToolCallEvent
from roomkit.tools.result import GateRefusal, pre_execution_denial
from roomkit.tools.validation import fold_hoisted_arguments, validate_tool_arguments

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.context import RoomContext
    from roomkit.tools.policy import ToolPolicy
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_voice")


class RealtimeToolGateMixin:
    """The catalogue, the policy and the pre-execution gate of a realtime session."""

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_tools: dict[str, Any]
    _tools: Any
    _skill_support: Any
    _tool_policy: ToolPolicy | None
    _session_roles: dict[str, str | None]
    _collisions: CollisionLog
    _registry: ChannelRegistry
    _tool_search_support: Any
    _provider: RealtimeVoiceProvider
    _framework: RoomKit | None
    channel_id: str

    def _session_base_tools(self, session_id: str) -> list[dict[str, Any]]:
        """The session's authorized catalogue, read under the state lock."""
        with self._state_lock:
            return self._session_tools.get(session_id, self._tools or [])

    def _session_catalogue(self, session_id: str) -> list[dict[str, Any]]:
        """Every tool the session declares beside the channel's own: its base
        catalogue, then what orchestration set up for its room (RFC §19.7).
        What a call is checked, validated and recovered against."""
        with self._state_lock:
            base = list(self._session_tools.get(session_id, self._tools or []))
            room_id = self._session_rooms.get(session_id)
        return base + self._orchestration_dicts(room_id, {dict_tool_name(t) for t in base})

    def _orchestration_dicts(
        self, room_id: str | None, skip: Container[str | None] = ()
    ) -> list[dict[str, Any]]:
        """The declarations of the tools orchestration declares in *room_id*'s
        sessions, bar the names in *skip*."""
        return [
            tool_dict(entry.definition)
            for entry in self._registry.entries(room_id, source=ToolSource.ORCHESTRATION)
            if entry.traits.always_declared and entry.name not in skip
        ]

    def _tool_parameters(self, name: str, session: VoiceSession) -> dict[str, Any] | None:
        """Return the declared ``parameters`` schema for realtime tool *name*.

        ``None`` when the tool's schema is unknown (skips argument validation).
        """
        if self._tool_search_support and self._tool_search_support.is_search_tool(name):
            for tool in self._tool_search_support.search_tool_dicts():
                if tool["name"] == name:
                    params = tool.get("parameters")
                    return params if isinstance(params, dict) else None
        if self._skill_support and self._skill_support.is_skill_tool(name):
            for tool in self._skill_support.skill_tool_dicts():
                if tool["name"] == name:
                    params = tool.get("parameters")
                    return params if isinstance(params, dict) else None
        for t in self._session_catalogue(session.id):
            if isinstance(t, dict) and t.get("name") == name:
                params = t.get("parameters")
                return params if isinstance(params, dict) else None
        return None

    def _is_declared_realtime_tool(
        self, name: str, session: VoiceSession, served: Container[str] | None = None
    ) -> bool:
        """Return whether *name* is in a non-empty session tool catalogue.

        An empty catalogue retains the historical hook-only/dynamic-handler
        mode. Once declarations exist, however, a provider cannot invent an
        undeclared name and reach a generic dispatcher.

        Infrastructure tools (Tool Search, skills) are declared by the channel
        rather than by the caller's catalogue, so they answer ``True`` without
        appearing in it.
        """
        if name in (self._channel_tool_names() if served is None else served):
            return True
        tools = self._session_catalogue(session.id)
        if not tools:
            return True
        return any(isinstance(tool, dict) and tool.get("name") == name for tool in tools)

    def _channel_tool_names(self) -> frozenset[str]:
        """The tools this channel serves itself: Tool Search's and the skills'."""
        return frozenset(e.name for e in self._registry.entries(None, source=ToolSource.CHANNEL))

    def _exempt_tool_names(self) -> frozenset[str]:
        """The channel's own tools that escape the policy and skill gating (RFC §21.1)."""
        return frozenset(self._registry.names(None, lambda traits: traits.exempt))

    def _declared_once(
        self, tools: list[dict[str, Any]], room_id: str | None
    ) -> list[dict[str, Any]]:
        """A session's host tools in *room_id*: none under a name the channel or
        orchestration declares itself there, each name once (RFC §21.1,
        :func:`declared_once`). Those are composed in afterwards."""
        served = self._channel_tool_names() | self._registry.names(
            room_id, lambda traits: traits.always_declared
        )
        return declared_once(tools, dict_tool_name, served, self._collisions)

    def _tool_reachable(self, name: str, session_id: str) -> bool:
        """Whether the session may call *name*: its tool policy and skill gating.

        What Tool Search may name in its results and listings (RFC §21.1); the
        pre-execution gate enforces the same rule on the call itself.
        """
        if not policy_admits(self._session_policy(session_id), name, self._exempt_tool_names()):
            return False
        support = self._skill_support
        return support is None or not support.is_gated(name, session_id)

    def _session_policy(self, session_id: str) -> ToolPolicy | None:
        """The tool policy resolved for the session's participant (RFC §12.4)."""
        if self._tool_policy is None:
            return None
        return self._tool_policy.resolve(self._session_roles.get(session_id))

    def _policy_filter(self, session_id: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The part of *tools* the session's policy admits.

        Tool Search's ``call_tool`` transport stays declared: it is no tool of
        its own, and the policy applies to the tool it names, at the gate.
        """
        policy = self._session_policy(session_id)
        if policy is None:
            return tools
        search = self._tool_search_support
        exempt = self._exempt_tool_names()
        return [
            t
            for t in tools
            if (search is not None and search.is_search_tool(str(t.get("name", ""))))
            or policy_admits(policy, str(t.get("name", "")), exempt)
        ]

    async def _refresh_session_role(self, session: VoiceSession, room_id: str | None) -> None:
        """Read the participant's role again, so a role changed during the
        session holds at the gate from the next call on (RFC §12.4)."""
        policy = self._tool_policy
        if policy is None or not policy.role_overrides or not (self._framework and room_id):
            return
        role = await self._resolve_session_role(room_id, session.participant_id)
        if session.id in self._session_roles:
            self._session_roles[session.id] = role

    async def _resolve_session_role(self, room_id: str | None, participant_id: str) -> str | None:
        """The session participant's role, where a policy has overrides to read."""
        policy = self._tool_policy
        if policy is None or not policy.role_overrides or not (self._framework and room_id):
            return None
        # Under the framework's lease, like every store read a channel makes:
        # a call landing while the kit closes must not read a closing store.
        with self._framework._resource_lease():
            participant = await self._framework.store.get_participant(room_id, participant_id)
        return participant.role if participant is not None else None

    async def _authorize_realtime_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        call_id: str,
        room_id: str | None,
        session: VoiceSession,
        *,
        channel_serves: bool = True,
    ) -> tuple[dict[str, Any], GateRefusal | None, RoomContext | None]:
        """Pre-execution gate for realtime tool calls (parity with the classic
        AI path), in RFC §12.4's order.

        *channel_serves* says whether this entry serves the channel's own
        tools (Tool Search, skills): the provider's function calls do; a
        reasoning backend's calls and a recovered spoken call reach the
        handler only, so on them no name is the channel's (RFC §21.1).

        Checks the tool is declared, folds a flattened hub-tool call back into
        ``params`` and validates the arguments against the declared schema,
        applies the tool policy and skill gating, and runs BEFORE_TOOL_USE so
        a block prevents the side effect rather than only hiding the result.
        Hooks may replace the arguments through ``metadata["arguments"]``; the
        replacement is validated before it can reach the handler.

        Returns the effective arguments, an optional denial result, and the
        room context this gate built — ``None`` when it built none. The caller
        hands that context to ON_TOOL_CALL's judgement as ``carrying`` so one
        tool call deserialises the room history once instead of twice.
        """
        served = self._channel_tool_names() if channel_serves else frozenset()
        if not self._is_declared_realtime_tool(name, session, served):
            logger.warning("Realtime provider requested undeclared tool %s", name)
            undeclared = json.dumps({"error": f"Tool '{name}' is not declared"})
            return arguments, GateRefusal(undeclared), None
        params = self._tool_parameters(name, session)
        arguments, invalid = self._validated_realtime_arguments(name, arguments, params)
        if invalid is not None:
            return arguments, GateRefusal(invalid), None
        await self._refresh_session_role(session, room_id)
        exempt = self._exempt_tool_names() if channel_serves else frozenset()
        refusal = self._access_refusal(name, session.id, exempt)
        if refusal is not None:
            return arguments, GateRefusal(refusal), None
        return await self._before_realtime_tool_use(
            name, arguments, params, call_id, room_id, session
        )

    def _validated_realtime_arguments(
        self, name: str, arguments: dict[str, Any], params: dict[str, Any] | None
    ) -> tuple[dict[str, Any], str | None]:
        """The model's arguments checked against the declared schema (fail-closed),
        after repairing a hub tool's flattened ``params``: same gate, same order
        as the classic AI path."""
        if params is None:
            return arguments, None
        folded, fold_error = fold_hoisted_arguments(params, arguments)
        if fold_error is not None:
            logger.warning("Realtime tool %s arguments ambiguous: %s", name, fold_error)
            return arguments, json.dumps(
                {"error": f"Invalid arguments for '{name}': {fold_error}"}
            )
        if folded is not None:
            logger.info(
                "Realtime tool %s: folded hoisted arguments %s into its container "
                "(provider=%s, model=%s)",
                name,
                sorted(set(arguments) - set(folded)),
                self._provider.name,
                self._provider.model_name,
            )
            arguments = folded
        arg_error = validate_tool_arguments(params, arguments)
        if arg_error is not None:
            logger.warning("Realtime tool %s arguments rejected: %s", name, arg_error)
            return arguments, json.dumps({"error": f"Invalid arguments for '{name}': {arg_error}"})
        return arguments, None

    def _access_refusal(self, name: str, session_id: str, exempt: Container[str]) -> str | None:
        """Why the session may not call *name*: its tool policy, resolved for
        its participant, then skill gating, as on the classic path."""
        if not policy_admits(self._session_policy(session_id), name, exempt):
            logger.warning("Realtime tool %s blocked by policy", name)
            return json.dumps({"error": policy_refusal(name)})
        # Hiding a gated tool from the catalogue is not enforcement — the model
        # may still name one it saw before the skill was deactivated.
        if self._skill_support is not None and self._skill_support.is_gated(name, session_id):
            logger.warning("Realtime tool %s blocked by skill gating", name)
            return json.dumps(
                {
                    "error": (
                        f"Tool '{name}' is gated by a skill. "
                        "Activate the skill first using activate_skill."
                    )
                }
            )
        return None

    async def _before_realtime_tool_use(
        self,
        name: str,
        arguments: dict[str, Any],
        params: dict[str, Any] | None,
        call_id: str,
        room_id: str | None,
        session: VoiceSession,
    ) -> tuple[dict[str, Any], GateRefusal | None, RoomContext | None]:
        """BEFORE_TOOL_USE as every channel runs it, which needs a framework and
        a room to run room hooks; the arguments it leaves are validated again."""
        framework = self._framework
        if framework is None or not room_id:
            return arguments, None, None
        pre_event = ToolCallEvent(
            channel_id=self.channel_id,
            channel_type=ChannelType.REALTIME_VOICE,
            tool_call_id=call_id,
            name=name,
            arguments=arguments,
            result=None,
            room_id=room_id,
            session=session,
        )
        decision, context = await framework._decide_before_tool_use(pre_event, self.channel_id)
        if not decision:
            logger.info("Realtime tool %s denied by BEFORE_TOOL_USE hook", name)
            denial = json.dumps({"error": pre_execution_denial(name, decision.reason)})
            return arguments, GateRefusal(denial, decision.detail), context
        arguments, invalid = _rewritten_arguments(name, arguments, params, decision.arguments)
        return arguments, GateRefusal(invalid) if invalid is not None else None, context


def _rewritten_arguments(
    name: str,
    arguments: dict[str, Any],
    params: dict[str, Any] | None,
    rewritten: dict[str, Any] | None,
) -> tuple[dict[str, Any], str | None]:
    """The arguments BEFORE_TOOL_USE left, returned or edited in place, checked
    against the schema again."""
    effective = rewritten if rewritten is not None else arguments
    # No fold here, deliberately: a hook's rewritten arguments are user code,
    # and repairing them would hide the hook's bug. The model's own call was
    # already folded above.
    arg_error = validate_tool_arguments(params, effective) if params is not None else None
    if arg_error is not None:
        logger.warning("Realtime tool %s post-hook arguments rejected: %s", name, arg_error)
        return effective, json.dumps(
            {"error": f"Invalid rewritten arguments for '{name}': {arg_error}"}
        )
    return effective, None
