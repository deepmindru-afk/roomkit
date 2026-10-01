"""A conference provider's tool calls, through the tool gate (RFC 12.10.12).

The conference is one more door to the host's tools, so a call through it
passes the same steps as through any other: the declared tool and its schema,
the tool policy, BEFORE_TOOL_USE, the handler, ON_TOOL_CALL (the SYNC chain, then its
observers), the bound on the result. Nothing here is a rule of its own: it
composes the pieces the AI channel's tool loop already runs on, the framework's
hook callbacks and ``roomkit.tools.validation``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from roomkit.channels._ai_policy import policy_admits, policy_refusal
from roomkit.channels._realtime_tools import result_text
from roomkit.channels._served_tools import CollisionLog, declared_once, dict_tool_name
from roomkit.core.exceptions import ToolRefusedError
from roomkit.models.enums import ChannelType, HookTrigger
from roomkit.models.tool_call import (
    ToolCallCallback,
    ToolCallEvent,
    ToolCallObserver,
)
from roomkit.tools.result import (
    GateRefusal,
    failure_detail,
    pre_execution_denial,
    read_tool_call_verdict,
    tool_failure,
)
from roomkit.tools.timeout import answer_within
from roomkit.tools.validation import fold_hoisted_arguments, validate_tool_arguments

if TYPE_CHECKING:
    from roomkit.conference.models import ConferenceRealtimeConfig
    from roomkit.core.framework import RoomKit
    from roomkit.core.hooks import HookEngine
    from roomkit.tools.external import BeforeToolCallback
    from roomkit.voice.base import VoiceSession

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


@dataclass(frozen=True)
class ToolOutcome:
    """What the gate and the handler made of one call, before ON_TOOL_CALL."""

    event: ToolCallEvent
    """The call, with the arguments it ran with (or would have)."""
    body: str
    """The handler's result, or the refusal the model reads."""
    served: bool


class ConferenceToolGate:
    """Serves one conference channel's tool calls; refusals reach observers only.

    A call is answered in three steps, because the caller sits between them:
    :meth:`execute` runs the gate and the handler, :meth:`result` runs
    ON_TOOL_CALL on a served call and bounds what the model reads, and
    :meth:`report_refusal` reports a call nothing served, once its refusal is
    on the wire (RFC 12.4).
    """

    def __init__(self, channel_id: str) -> None:
        self._channel_id = channel_id
        self._hooks: HookEngine | None = None
        self._before: BeforeToolCallback | None = None
        self._served: ToolCallCallback | None = None
        self._observed: ToolCallObserver | None = None

    def set_framework(self, framework: RoomKit) -> None:
        self._hooks = framework.hook_engine
        self._before = framework._build_before_tool_call_hook(self._channel_id)
        self._served = framework._build_tool_call_hook(self._channel_id)
        self._observed = framework._build_tool_observer_hook(self._channel_id)

    def event(
        self, session: VoiceSession, call_id: str, name: str, arguments: dict[str, Any]
    ) -> ToolCallEvent:
        """The ON_TOOL_CALL event of one call of *session*."""
        return ToolCallEvent(
            channel_id=self._channel_id,
            channel_type=ChannelType.CONFERENCE,
            tool_call_id=call_id,
            name=name,
            arguments=arguments,
            room_id=session.room_id,
            session=session,
        )

    async def execute(self, config: ConferenceRealtimeConfig, event: ToolCallEvent) -> ToolOutcome:
        """Run *event*'s call through the gate, then the handler."""
        arguments, denial = await self._authorize(config, event)
        event = replace(event, arguments=arguments)
        if denial is not None:
            return self._refused(replace(event, error_detail=denial.detail), _error(denial.body))
        if config.tool_handler is None:
            reason = f"no handler is configured for tool {event.name!r}"
            return self._refused(event, _error(reason))
        try:
            timeout = config.tool_bound(event.name)
            answer = config.tool_handler(str(event.room_id), event.name, arguments)
            result = await answer_within(timeout, event.name, answer)
        except ToolRefusedError as refusal:
            # A declined call, in the handler's own words.
            return self._refused(event, refusal.message)
        except Exception as exc:
            logger.exception(
                "Conference channel %r: the tool handler failed on %r in room %s",
                self._channel_id,
                event.name,
                event.room_id,
            )
            return self.failure(event, exc)
        return ToolOutcome(event, result_text(result), served=True)

    def failure(self, event: ToolCallEvent, exc: Exception) -> ToolOutcome:
        """The outcome of a call that raised: the model reads its class, the
        message rides the event for the observers (RFC §9.3)."""
        detailed = replace(event, error_detail=failure_detail(exc))
        return ToolOutcome(detailed, tool_failure(event.name, exc), served=False)

    async def result(self, outcome: ToolOutcome) -> str:
        """What the model reads: a served result once ON_TOOL_CALL ran on it
        (the SYNC chain, then its observers), a refusal as it is; bounded."""
        body = outcome.body
        if outcome.served and self._served is not None:
            verdict = await self._served(replace(outcome.event, result=body))
            # Read as on every channel: a block withholds the result, and a
            # hook's result, a bare one included, replaces it (RFC §9.3).
            body = result_text(read_tool_call_verdict(outcome.event.name, verdict, body).result)
        return _bounded(body, outcome.event.name)

    async def report_refusal(
        self, event: ToolCallEvent, body: str, *, cancelled: bool = False
    ) -> None:
        """Report a call nothing served to ON_TOOL_CALL's observers."""
        if self._observed is not None:
            await self._observed(replace(event, result=body, is_error=True, cancelled=cancelled))

    def _refused(self, event: ToolCallEvent, body: str) -> ToolOutcome:
        logger.info(
            "Conference channel %r refused tool %r in room %s: %s",
            self._channel_id,
            event.name,
            event.room_id,
            body,
        )
        return ToolOutcome(event, body, served=False)

    async def _authorize(
        self, config: ConferenceRealtimeConfig, event: ToolCallEvent
    ) -> tuple[dict[str, Any], GateRefusal | None]:
        """The pre-execution gate (RFC 12.4): the arguments to run with, or why not."""
        name, arguments = event.name, event.arguments
        declared = {t.get("name"): t for t in config.tools or [] if isinstance(t, dict)}
        if declared and name not in declared:
            logger.warning("Conference provider requested undeclared tool %s", name)
            return arguments, GateRefusal(f"Tool '{name}' is not declared")
        params = declared.get(name, {}).get("parameters")
        schema = params if isinstance(params, dict) else None
        if schema is not None:
            folded, fold_error = fold_hoisted_arguments(schema, arguments)
            arguments = folded if folded is not None else arguments
            error = fold_error or validate_tool_arguments(schema, arguments)
            if error is not None:
                return arguments, GateRefusal(f"Invalid arguments for '{name}': {error}")
        if not policy_admits(config.tool_policy, name, _NO_CHANNEL_TOOLS):
            logger.warning("Conference tool %s blocked by policy", name)
            return arguments, GateRefusal(policy_refusal(name))
        return await self._before_tool_use(event, arguments, schema)

    async def _before_tool_use(
        self, event: ToolCallEvent, arguments: dict[str, Any], schema: dict[str, Any] | None
    ) -> tuple[dict[str, Any], GateRefusal | None]:
        """BEFORE_TOOL_USE, when a hook listens; what it leaves is validated again.

        A hook may return new arguments or edit the event's own in place:
        either way the handler must not run on arguments the schema rejects.
        """
        if self._before is None or self._hooks is None:
            return arguments, None
        if not self._hooks.has_hooks(HookTrigger.BEFORE_TOOL_USE):
            return arguments, None
        name = event.name
        decision = await self._before(replace(event, arguments=arguments))
        if not decision:
            return arguments, GateRefusal(pre_execution_denial(name), decision.detail)
        if decision.arguments is not None:
            arguments = decision.arguments
        error = validate_tool_arguments(schema, arguments) if schema is not None else None
        if error is not None:
            return arguments, GateRefusal(f"Invalid rewritten arguments for '{name}': {error}")
        return arguments, None


def _error(reason: str) -> str:
    return json.dumps({"error": reason})


def _bounded(text: str, name: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    logger.warning(
        "Conference tool result for %s truncated from %d to %d chars",
        name,
        len(text),
        MAX_RESULT_CHARS,
    )
    notice = f"\n... [truncated: the result was {len(text)} characters]"
    return text[: MAX_RESULT_CHARS - len(notice)] + notice
