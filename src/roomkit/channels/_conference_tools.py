"""A conference provider's tool calls: their pre-execution gate (RFC 12.10.12).

The conference is one more door to the host's tools, so a call through it
passes the same steps as through any other, run by the realtime tool executor:
this gate (the declared tool and its schema, the tool policy, BEFORE_TOOL_USE),
the handler, ON_TOOL_CALL, the bound on the result. Nothing here is a rule of
its own: the gate composes the kit's BEFORE_TOOL_USE decision and
``roomkit.tools.validation``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from roomkit.channels._ai_policy import policy_admits, policy_refusal
from roomkit.channels._served_tools import CollisionLog, declared_once, dict_tool_name
from roomkit.models.enums import ChannelType
from roomkit.models.tool_call import ToolCallEvent
from roomkit.tools.result import GateRefusal, bounded_result, pre_execution_denial
from roomkit.tools.validation import fold_hoisted_arguments, validate_tool_arguments

if TYPE_CHECKING:
    from roomkit.channels._realtime_tool_calls import RealtimeToolCall
    from roomkit.conference.models import ConferenceRealtimeConfig
    from roomkit.core.framework import RoomKit
    from roomkit.tools.external import BeforeToolCallback

logger = logging.getLogger("roomkit.channels.conference")

MAX_RESULT_CHARS = 16384
"""The bound on a result the model reads: RealtimeVoiceChannel's default."""


#: A conference serves no tool of its own: the policy governs every name
#: (RFC §21.1).
_NO_CHANNEL_TOOLS: frozenset[str] = frozenset()


def declared_tools(
    config: ConferenceRealtimeConfig, collisions: CollisionLog
) -> list[dict[str, Any]] | None:
    """The tools a conference declares to its provider: what its policy admits,
    each name once, the later definition kept as its gate reads it (RFC §21.1)."""
    if config.tools is None:
        return None
    tools = declared_once(config.tools, dict_tool_name, _NO_CHANNEL_TOOLS, collisions)
    return [
        t
        for t in tools
        if policy_admits(config.tool_policy, str(t.get("name", "")), _NO_CHANNEL_TOOLS)
    ]


def warn_unused_role_overrides(config: ConferenceRealtimeConfig, channel_id: str) -> None:
    """Log that a conference policy's role overrides never apply: the mix
    names no participant, so only the base rules do."""
    policy = config.tool_policy
    if policy is not None and policy.role_overrides:
        logger.warning(
            "Conference channel %r: tool_policy role_overrides %s never apply; "
            "the mix names no participant, so only the base rules do",
            channel_id,
            sorted(policy.role_overrides),
        )


class ConferenceToolGate:
    """The pre-execution gate of one conference channel's tool calls (RFC 12.4)."""

    def __init__(self, channel_id: str) -> None:
        self._channel_id = channel_id
        self._before: BeforeToolCallback | None = None

    def set_framework(self, framework: RoomKit) -> None:
        self._before = framework._build_before_tool_call_hook(self._channel_id)

    def event(self, call: RealtimeToolCall, result: str | None = None) -> ToolCallEvent:
        """The ON_TOOL_CALL event of one call of a conference session."""
        return ToolCallEvent(
            channel_id=self._channel_id,
            channel_type=ChannelType.CONFERENCE,
            tool_call_id=call.call_id,
            name=call.name,
            arguments=call.arguments,
            result=result,
            room_id=call.room_id,
            session=call.session,
        )

    async def authorize(
        self, config: ConferenceRealtimeConfig, call: RealtimeToolCall
    ) -> GateRefusal | None:
        """The pre-execution gate (RFC 12.4): why *call* may not run; the
        arguments to run with are left on *call*."""
        name = call.name
        declared = {t.get("name"): t for t in config.tools or [] if isinstance(t, dict)}
        if declared and name not in declared:
            logger.warning("Conference provider requested undeclared tool %s", name)
            return GateRefusal(_error(f"Tool '{name}' is not declared"))
        params = declared.get(name, {}).get("parameters")
        schema = params if isinstance(params, dict) else None
        if schema is not None:
            folded, fold_error = fold_hoisted_arguments(schema, call.arguments)
            call.arguments = folded if folded is not None else call.arguments
            error = fold_error or validate_tool_arguments(schema, call.arguments)
            if error is not None:
                return GateRefusal(_error(f"Invalid arguments for '{name}': {error}"))
        if not policy_admits(config.tool_policy, name, _NO_CHANNEL_TOOLS):
            logger.warning("Conference tool %s blocked by policy", name)
            return GateRefusal(_error(policy_refusal(name)))
        return await self._before_tool_use(call, schema)

    async def _before_tool_use(
        self, call: RealtimeToolCall, schema: dict[str, Any] | None
    ) -> GateRefusal | None:
        """BEFORE_TOOL_USE, as every channel runs it; what it leaves is
        validated again.

        A hook may return new arguments or edit the event's own in place:
        either way the handler must not run on arguments the schema rejects.
        """
        if self._before is None:
            return None
        name = call.name
        decision = await self._before(self.event(call))
        if not decision:
            denial = pre_execution_denial(name, decision.reason)
            return GateRefusal(_error(denial), decision.detail)
        if decision.arguments is not None:
            call.arguments = decision.arguments
        error = validate_tool_arguments(schema, call.arguments) if schema is not None else None
        if error is not None:
            return GateRefusal(_error(f"Invalid rewritten arguments for '{name}': {error}"))
        return None


def bound_result(text: str, name: str) -> str:
    """*text* within the bound on a result the model reads (RFC §21.5)."""
    return bounded_result(text, MAX_RESULT_CHARS, name)


def _error(reason: str) -> str:
    return json.dumps({"error": reason})
