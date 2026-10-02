"""Lifecycle of the tool calls a streaming provider serves itself.

A call the provider already ran (its result rides it) is reported, and a call
its external tool handler decides is decided, then reported. Every other call
is the channel's own, served by the loop (RFC §9.3, who serves a call).
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from roomkit.models.enums import ChannelType
from roomkit.models.streaming import StreamDelta, ToolCallEndMarker, ToolCallStartMarker
from roomkit.models.tool_call import ToolCallEvent, ToolCallObserver
from roomkit.providers.ai.base import StreamToolCall
from roomkit.providers.ai.tool_calls import cut_call_error
from roomkit.realtime.base import EphemeralEventType
from roomkit.tools._outcome import OutcomeKind, ToolOutcome
from roomkit.tools.external import ExternalToolHandler
from roomkit.tools.result import as_tool_result


class _ToolEventPublisher(Protocol):
    async def __call__(
        self,
        event_type: EphemeralEventType,
        room_id: str,
        tool_calls: list[Any],
        round_idx: int,
        *,
        duration_ms: int | None = None,
    ) -> None: ...


@dataclass
class _ExternalStreamTools:
    """Turn-scoped routing and lifecycle of the calls the provider serves."""

    channel_id: str
    room_id: str | None
    publish: _ToolEventPublisher
    # Whether the turn has a tool of the channel's own under a name.
    serves_locally: Callable[[str], bool]
    handler: ExternalToolHandler | None = None
    # ON_TOOL_CALL as a report, for a call the provider already ran (RFC §9.3).
    report: ToolCallObserver | None = None

    def takes(self, call: StreamToolCall) -> bool:
        """Whether *call* is the provider's: one it already ran, or one its
        handler decides because no tool of the channel's own carries its name."""
        if "_result" in call.arguments:
            return True
        return self.handler is not None and not self.serves_locally(call.name)

    async def stream_call(
        self, call: StreamToolCall, round_idx: int
    ) -> AsyncGenerator[StreamDelta, None]:
        """Report a call the provider serves inline, keeping persistence markers
        around its callbacks."""
        arguments = dict(call.arguments)
        already_executed = "_result" in arguments
        result = arguments.pop("_result", None) or ""
        is_error = arguments.pop("_is_error", False)

        yield ToolCallStartMarker(tool_name=call.name, tool_id=call.id, arguments=arguments)
        if self.room_id:
            await self.publish(
                EphemeralEventType.TOOL_CALL_START,
                self.room_id,
                [call.model_copy(update={"arguments": arguments})],
                round_idx,
            )

        started_at = time.monotonic()
        # A proxy's embedded result means the side effect already happened.
        # Only a still-pending call can be denied or rewritten before acting.
        if not already_executed and self.handler is not None:
            arguments, result, is_error = await self._decide(
                self.handler, call, arguments, result, bool(is_error)
            )
        await self._report(call, arguments, result, bool(is_error))

        duration_ms = int((time.monotonic() - started_at) * 1000)
        yield ToolCallEndMarker(
            tool_name=call.name,
            tool_id=call.id,
            arguments=arguments,
            result=result,
            status="failed" if is_error else "completed",
            duration_ms=duration_ms,
            error=result if is_error else None,
        )
        if self.room_id:
            await self.publish(
                EphemeralEventType.TOOL_CALL_END,
                self.room_id,
                [ToolOutcome(_external_kind(is_error), result).as_part(call.id, call.name)],
                round_idx,
                duration_ms=duration_ms,
            )

    async def _decide(
        self,
        handler: ExternalToolHandler,
        call: StreamToolCall,
        arguments: dict[str, Any],
        result: str,
        is_error: bool,
    ) -> tuple[dict[str, Any], str, bool]:
        """What a still-pending call becomes: its arguments, result and error flag.

        A call the response cut before its arguments were complete is refused
        without asking the handler (RFC §6.4); any other is the handler's to
        deny, rewrite or serve.
        """
        if call.partial:
            return arguments, json.dumps(cut_call_error(call.name)), True
        decision = await handler.process_tool_call(
            call.name, arguments, tool_call_id=call.id, room_id=self.room_id
        )
        if not decision.approved:
            return (
                arguments,
                json.dumps({"error": decision.reason or f"Tool '{call.name}' was denied"}),
                True,
            )
        if decision.modified_input is not None:
            arguments = decision.modified_input
        if decision.result is not None:
            return arguments, decision.result, False
        return arguments, result, is_error

    async def _report(
        self, call: StreamToolCall, arguments: dict[str, Any], result: str, is_error: bool
    ) -> None:
        """Hand the call's outcome to its handler, or report it to ON_TOOL_CALL's
        observers when the provider ran it with no handler: an outcome the
        model already read, so no hook may rewrite it (RFC §9.3)."""
        if self.handler is not None:
            await self.handler.on_tool_result(
                call.name,
                arguments,
                result,
                is_error=is_error,
                tool_call_id=call.id,
                room_id=self.room_id,
            )
            return
        if self.report is None:
            return
        await self.report(
            ToolCallEvent(
                channel_id=self.channel_id,
                channel_type=ChannelType.AI,
                tool_call_id=call.id,
                name=call.name,
                arguments=arguments,
                result=as_tool_result(result),
                room_id=self.room_id,
                is_error=is_error,
            )
        )


def _external_kind(is_error: bool) -> OutcomeKind:
    """How a call the provider or the external handler ran ended, as it reported."""
    return OutcomeKind.FAILED if is_error else OutcomeKind.SERVED
