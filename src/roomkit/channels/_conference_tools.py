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
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from roomkit.channels._ai_policy import policy_admits, policy_refusal
from roomkit.channels._served_tools import CollisionLog, declared_once, dict_tool_name
from roomkit.core.exceptions import ToolRefusedError, UnservedToolCallError
from roomkit.models.enums import ChannelType
from roomkit.models.tool_call import (
    ToolCallCallback,
    ToolCallEvent,
    ToolCallObserver,
    ToolCallVerdict,
)
from roomkit.tools._outcome import OutcomeKind, ToolOutcome, read_outcome
from roomkit.tools.result import (
    GateRefusal,
    declined_answer,
    failure_detail,
    pre_execution_denial,
    read_tool_call_verdict,
    result_text,
    tool_failure,
    unserved_tool_error,
)
from roomkit.tools.timeout import answer_within
from roomkit.tools.validation import fold_hoisted_arguments, validate_tool_arguments

if TYPE_CHECKING:
    from roomkit.conference.models import ConferenceRealtimeConfig
    from roomkit.core.framework import RoomKit
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
        self._before: BeforeToolCallback | None = None
        self._served: ToolCallCallback | None = None
        self._observed: ToolCallObserver | None = None

    def set_framework(self, framework: RoomKit) -> None:
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

    async def execute(
        self, config: ConferenceRealtimeConfig, event: ToolCallEvent
    ) -> tuple[ToolCallEvent, ToolOutcome]:
        """Run *event*'s call through the gate, then the handler: the call with
        the arguments it ran with, and its outcome before ON_TOOL_CALL."""
        arguments, denial = await self._authorize(config, event)
        event = replace(event, arguments=arguments)
        if denial is not None:
            return self._refused(event, _error(denial.body), denial.detail)
        if config.tool_handler is None:
            reason = f"no handler is configured for tool {event.name!r}"
            return self._refused(event, _error(reason))
        try:
            timeout = config.tool_bound(event.name)
            answer = config.tool_handler(str(event.room_id), event.name, arguments)
            result = declined_answer(await answer_within(timeout, event.name, answer), event.name)
        except UnservedToolCallError:
            # Not the handler's to serve: the hooks may still (RFC §21.4).
            return event, ToolOutcome(OutcomeKind.UNSERVED, unserved_tool_error(event.name))
        except ToolRefusedError as refusal:
            # A refused call, in the handler's own words.
            return self._refused(event, refusal.message)
        except Exception as exc:
            logger.exception(
                "Conference channel %r: the tool handler failed on %r in room %s",
                self._channel_id,
                event.name,
                event.room_id,
            )
            return self.failure(event, exc)
        return event, ToolOutcome(OutcomeKind.SERVED, result_text(result))

    def failure(self, event: ToolCallEvent, exc: Exception) -> tuple[ToolCallEvent, ToolOutcome]:
        """The outcome of a call that raised: the model reads its class, the
        message rides the outcome for the observers (RFC §9.3)."""
        body = tool_failure(event.name, exc)
        return event, ToolOutcome(OutcomeKind.FAILED, body, detail=failure_detail(exc))

    async def result(self, event: ToolCallEvent, outcome: ToolOutcome) -> ToolOutcome:
        """The call's final outcome, bounded for the model: ON_TOOL_CALL's
        verdict on a served call (the SYNC chain, then its observers), the
        hooks' chance to serve a call nothing served, a refusal as it is."""
        if outcome.kind in _JUDGED and self._served is not None:
            served = outcome.result if outcome.kind is OutcomeKind.SERVED else None
            verdict = await self._served(replace(event, result=served))
            # Read as on every channel: a block withholds the result, and a
            # hook's result, a bare one included, replaces it (RFC §9.3).
            reading = read_tool_call_verdict(event.name, verdict, served)
            detail = verdict.error_detail if isinstance(verdict, ToolCallVerdict) else None
            outcome = ToolOutcome(read_outcome(reading), reading.result, detail=detail)
        return replace(outcome, result=_bounded(result_text(outcome.result), event.name))

    async def report_refusal(
        self,
        event: ToolCallEvent,
        body: str,
        *,
        detail: str | None = None,
        cancelled: bool = False,
    ) -> None:
        """Report a call nothing served to ON_TOOL_CALL's observers."""
        if self._observed is not None:
            reported = replace(event, result=body, is_error=True, cancelled=cancelled)
            if detail is not None:
                reported = replace(reported, error_detail=detail)
            await self._observed(reported)

    def _refused(
        self, event: ToolCallEvent, body: str, detail: str | None = None
    ) -> tuple[ToolCallEvent, ToolOutcome]:
        logger.info(
            "Conference channel %r refused tool %r in room %s: %s",
            self._channel_id,
            event.name,
            event.room_id,
            body,
        )
        return event, ToolOutcome(OutcomeKind.REFUSED, body, detail=detail)

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
        if self._before is None:
            return arguments, None
        name = event.name
        decision = await self._before(replace(event, arguments=arguments))
        if not decision:
            denial = pre_execution_denial(name, decision.reason)
            return arguments, GateRefusal(denial, decision.detail)
        if decision.arguments is not None:
            arguments = decision.arguments
        error = validate_tool_arguments(schema, arguments) if schema is not None else None
        if error is not None:
            return arguments, GateRefusal(f"Invalid rewritten arguments for '{name}': {error}")
        return arguments, None


# The outcomes ON_TOOL_CALL's SYNC chain reads: a served call, and one nothing
# served, which a hook may serve.
_JUDGED = frozenset({OutcomeKind.SERVED, OutcomeKind.UNSERVED})

# The outcomes the gate reports itself; the chain observes the others.
REPORTED_BY_GATE = frozenset({OutcomeKind.REFUSED, OutcomeKind.FAILED, OutcomeKind.UNSERVED})


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
