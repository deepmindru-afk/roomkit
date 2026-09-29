"""Lifecycle of tool calls served by an external streaming provider."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from roomkit.models.enums import ChannelType
from roomkit.models.streaming import StreamDelta, ToolCallEndMarker, ToolCallStartMarker
from roomkit.models.tool_call import ToolCallCallback, ToolCallEvent
from roomkit.providers.ai.base import AIToolResultPart, StreamToolCall
from roomkit.providers.ai.tool_calls import cut_call_error
from roomkit.realtime.base import EphemeralEventType
from roomkit.tools.external import BeforeToolCallback, ExternalToolHandler
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
    """Turn-scoped callbacks for externally served calls, with no local dispatch."""

    channel_id: str
    room_id: str | None
    publish: _ToolEventPublisher
    handler: ExternalToolHandler | None = None
    before: BeforeToolCallback | None = None
    after: ToolCallCallback | None = None

    async def stream_call(
        self, call: StreamToolCall, round_idx: int
    ) -> AsyncGenerator[StreamDelta, None]:
        """Observe a call inline, keeping persistence markers around its callbacks."""
        handler = self.handler
        if handler is None:
            return
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
        if not already_executed:
            arguments, result, is_error = await self._decide(
                handler, call, arguments, result, bool(is_error)
            )
        await handler.on_tool_result(
            call.name,
            arguments,
            result,
            is_error=bool(is_error),
            tool_call_id=call.id,
            room_id=self.room_id,
        )

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
                [AIToolResultPart(tool_call_id=call.id, name=call.name, result=result)],
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

    async def observe_calls(self, calls: Sequence[StreamToolCall]) -> None:
        """Notify hooks after the round when no external handler served it inline."""
        if self.handler is not None:
            return
        for call in calls:
            arguments = dict(call.arguments)
            already_executed = "_result" in arguments
            result = arguments.pop("_result", None)
            # The proxy's own verdict on a call it already ran. It travelled
            # this far as a private argument; drop it from what the hook reads
            # as arguments, keep it as the outcome it is.
            is_error = bool(arguments.pop("_is_error", False))
            event = ToolCallEvent(
                channel_id=self.channel_id,
                channel_type=ChannelType.AI,
                tool_call_id=call.id,
                name=call.name,
                arguments=arguments,
                # A call the proxy already ran has an outcome, empty when it
                # sent none (as in stream_call): None would read as a call
                # nothing served (RFC §9.3).
                result=(
                    ("" if result is None else as_tool_result(result))
                    if already_executed
                    else None
                ),
                room_id=self.room_id,
                is_error=is_error,
            )
            if already_executed and self.after is not None:
                await self.after(event)
            elif not already_executed and self.before is not None:
                await self.before(event)
