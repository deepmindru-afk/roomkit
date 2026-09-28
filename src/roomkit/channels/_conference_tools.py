"""A conference provider's tool calls, through the tool gate (RFC 12.10.12).

The conference is one more door to the host's tools, so a call through it
passes the same steps as through any other: the declared tool and its schema,
BEFORE_TOOL_USE, the handler, ON_TOOL_CALL (the SYNC chain, then its
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
from roomkit.core.exceptions import ToolRefusedError
from roomkit.models.enums import ChannelType
from roomkit.models.tool_call import ToolCallEvent, ToolCallVerdict
from roomkit.tools.validation import fold_hoisted_arguments, validate_tool_arguments

if TYPE_CHECKING:
    from roomkit.conference.models import ConferenceRealtimeConfig
    from roomkit.core.framework import RoomKit
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.channels.conference")

MAX_RESULT_CHARS = 16384
"""The bound on a result the model reads: RealtimeVoiceChannel's default."""


def declared_tools(config: ConferenceRealtimeConfig) -> list[dict[str, Any]] | None:
    """The tools a conference declares to its provider: what its policy admits."""
    if config.tools is None:
        return None
    return [t for t in config.tools if policy_admits(config.tool_policy, str(t.get("name", "")))]


class ConferenceToolGate:
    """Serves one conference channel's tool calls; refusals reach observers only."""

    def __init__(self, channel_id: str) -> None:
        self._channel_id = channel_id
        self._before: Any = None
        self._served: Any = None
        self._observed: Any = None

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

    async def answer(self, config: ConferenceRealtimeConfig, event: ToolCallEvent) -> str:
        """The result the model reads for *event*'s call, gate to bound."""
        arguments, denial = await self._authorize(config, event)
        if denial is not None:
            return await self.refuse(event, _error(denial))
        event = replace(event, arguments=arguments)
        if config.tool_handler is None:
            return await self.refuse(
                event, _error(f"no handler is configured for tool {event.name!r}")
            )
        try:
            result = await config.tool_handler(str(event.room_id), event.name, arguments)
        except ToolRefusedError as refusal:
            # A declined call, in the handler's own words.
            return await self.refuse(event, refusal.message)
        except Exception:
            logger.exception(
                "Conference channel %r: the tool handler failed on %r in room %s",
                self._channel_id,
                event.name,
                event.room_id,
            )
            # The exception is for the log: its text is no answer for a model.
            return await self.refuse(event, _error(f"Tool {event.name!r} failed"))
        return await self._serve(event, _as_text(result))

    async def refuse(self, event: ToolCallEvent, body: str, *, cancelled: bool = False) -> str:
        """Report a call that was not served to ON_TOOL_CALL's observers; the
        body the model reads is returned, bounded."""
        if self._observed is not None:
            await self._observed(replace(event, result=body, is_error=True, cancelled=cancelled))
        return _bounded(body, event.name)

    async def _authorize(
        self, config: ConferenceRealtimeConfig, event: ToolCallEvent
    ) -> tuple[dict[str, Any], str | None]:
        """The pre-execution gate (RFC 12.4): the arguments to run with, or why not."""
        name, arguments = event.name, event.arguments
        declared = {t.get("name"): t for t in config.tools or [] if isinstance(t, dict)}
        if declared and name not in declared:
            logger.warning("Conference provider requested undeclared tool %s", name)
            return arguments, f"Tool '{name}' is not declared"
        params = declared.get(name, {}).get("parameters")
        schema = params if isinstance(params, dict) else None
        if not policy_admits(config.tool_policy, name):
            logger.warning("Conference tool %s blocked by policy", name)
            return arguments, policy_refusal(name)
        if schema is not None:
            folded, fold_error = fold_hoisted_arguments(schema, arguments)
            arguments = folded if folded is not None else arguments
            error = fold_error or validate_tool_arguments(schema, arguments)
            if error is not None:
                return arguments, f"Invalid arguments for '{name}': {error}"
        if self._before is None:
            return arguments, None
        decision = await self._before(replace(event, arguments=arguments))
        if not decision:
            return arguments, f"Tool '{name}' denied by pre-execution hook."
        if decision.arguments is None:
            return arguments, None
        error = validate_tool_arguments(schema, decision.arguments) if schema else None
        if error is not None:
            return arguments, f"Invalid rewritten arguments for '{name}': {error}"
        return decision.arguments, None

    async def _serve(self, event: ToolCallEvent, result: str) -> str:
        """Run ON_TOOL_CALL on a served result; what the model reads, bounded."""
        if self._served is not None:
            verdict = await self._served(replace(event, result=result))
            if isinstance(verdict, ToolCallVerdict) and verdict.result is not None:
                result = _as_text(verdict.result)
        return _bounded(result, event.name)


def _error(reason: str) -> str:
    return json.dumps({"error": reason})


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


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
