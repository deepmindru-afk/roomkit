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
from roomkit.realtime.base import EphemeralEventType
from roomkit.tools.external import BeforeToolCallback, ExternalToolHandler


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
            decision = await handler.process_tool_call(
                call.name, arguments, tool_call_id=call.id, room_id=self.room_id
            )
            if not decision.approved:
                result = json.dumps({"error": decision.reason or f"Tool '{call.name}' was denied"})
                is_error = True
            else:
                if decision.modified_input is not None:
                    arguments = decision.modified_input
                if decision.result is not None:
                    result = decision.result
                    is_error = False
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

    async def observe_calls(self, calls: Sequence[StreamToolCall]) -> None:
        """Notify hooks after the round when no external handler served it inline."""
        if self.handler is not None:
            return
        for call in calls:
            arguments = dict(call.arguments)
            already_executed = "_result" in arguments
            result = arguments.pop("_result", None)
            arguments.pop("_is_error", None)
            event = ToolCallEvent(
                channel_id=self.channel_id,
                channel_type=ChannelType.AI,
                tool_call_id=call.id,
                name=call.name,
                arguments=arguments,
                result=(
                    result
                    if isinstance(result, (str, list))
                    else json.dumps(result)
                    if result is not None
                    else None
                ),
                room_id=self.room_id,
            )
            if already_executed and self.after is not None:
                await self.after(event)
            elif not already_executed and self.before is not None:
                await self.before(event)
