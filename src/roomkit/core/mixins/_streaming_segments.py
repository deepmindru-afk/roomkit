"""The writer of a streamed turn's timeline rows."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_event_result import tool_event_payload
from roomkit.models.enums import EventStatus, EventType, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.streaming import LoopEndMarker, ToolCallEndMarker, ToolCallStartMarker

if TYPE_CHECKING:
    from roomkit.core.event_router import StreamingResponse
    from roomkit.core.hooks import SyncPipelineResult
    from roomkit.core.lanes import DeliveryCascade
    from roomkit.core.mixins.lane_execution import DeliverySource
    from roomkit.models.context import RoomContext

logger = logging.getLogger("roomkit.inbound")


class SegmentWriter:
    """The one writer of a streamed turn's timeline rows.

    A stream produces three kinds of row — a text segment, a tool call's
    start, its end — and each used to build its event and decide for itself
    whether it crossed the ``BEFORE_BROADCAST`` hooks. The text did and the
    two markers did not, so a hook's decision (a display label, a PII
    rewrite, a refusal) never reached a stored tool row. They are verbs of
    one writer here, and the gate is the writer's own step: a fourth kind of
    row cannot be written past it.

    Each verb returns the row it committed, or ``None`` when there was
    nothing to write, a hook refused it, or the persistence policy excluded
    it (RFC §14.3) — which is exactly what the caller yields into the stream.
    """

    def __init__(
        self,
        kit: Any,
        sr: StreamingResponse,
        *,
        room_id: str,
        context: RoomContext,
        cascade: DeliveryCascade,
        plan_source: DeliverySource | str,
        chain_depth: int,
        visibility: str,
        correlation_id: str,
        parent_event_id: str | None,
        streamed_to: set[str],
        response_events: list[RoomEvent] | None,
    ) -> None:
        self._kit = kit
        self._sr = sr
        self._room_id = room_id
        self._context = context
        self._cascade = cascade
        self._plan_source = plan_source
        self._chain_depth = chain_depth
        self._visibility = visibility
        self._correlation_id = correlation_id
        self._parent_event_id = parent_event_id
        # The channel a text segment reaches *as it is produced* — the stream
        # itself is its delivery, so the lane must not send it again. Only the
        # first target streams (V1); any other streaming-capable channel is an
        # ordinary recipient.
        self._streamed_to = streamed_to
        self._response_events = response_events
        # The turn's record (``loop_end_reason``, ``ai_usage``), once its
        # ``LoopEndMarker`` is read; ``_record_owed`` while no row carries it.
        self._turn_record: dict[str, Any] | None = None
        self._record_owed = False
        self._accumulated: list[str] = []
        self._writing: set[asyncio.Task[RoomEvent | None]] = set()
        self._started: set[str] = set()
        self.persisted: list[RoomEvent] = []

    # -- what the stream hands in ------------------------------------------

    def add_text(self, delta: str) -> None:
        """Accumulate a text delta; a segment is written when something ends it."""
        self._accumulated.append(delta)

    def end_turn(self, marker: LoopEndMarker) -> None:
        """The loop reached its end: its record rides the turn's last message.

        The marker comes last, so the next flush is the turn's final text and
        carries it; a turn with no final text has it written on the message it
        already wrote, by :meth:`record_on_last_message` (RFC §6.4).
        """
        self._turn_record = {"ai_usage": dict(marker.usage), "loop_end_reason": marker.reason}
        self._record_owed = True

    async def record_on_last_message(self) -> None:
        """Write a record no final text carried on the last MESSAGE already stored.

        Run once the turn's deliveries are done, so no delivery record is
        written to the same row meanwhile. Only the row's metadata changes,
        through ``update_event``, so ON_EVENT_UPDATED sees it; it is not
        delivered again. Best effort: the turn's outcome stands without it.
        """
        if not self._record_owed or self._turn_record is None:
            return
        self._record_owed = False
        last = next((e for e in reversed(self.persisted) if e.type == EventType.MESSAGE), None)
        if last is None:
            return
        try:
            stored = await self._kit.store.get_event(last.id)
            if stored is None:
                return
            updated = await self._kit.update_event(
                self._room_id, last.id, metadata={**stored.metadata, **self._turn_record}
            )
        except Exception:
            logger.warning(
                "Could not record the turn's end on message %s (room %s)",
                last.id,
                self._room_id,
                exc_info=True,
            )
            return
        if updated is None:
            return
        for rows in (self.persisted, self._response_events):
            if rows is not None and last in rows:
                rows[rows.index(last)] = updated

    def stream_lost(self) -> None:
        """The stream failed: text accumulated past the failure never reached
        the streaming channel, so it goes out like any other event."""
        self._streamed_to.clear()

    # -- the three rows ----------------------------------------------------

    async def flush_text(self, *, cancelled: bool = False) -> RoomEvent | None:
        """Write the accumulated text as a MESSAGE row.

        ``sr.response_metadata`` (the turn's ``AIContext.response_metadata``,
        the same live record) rides every MESSAGE segment as it stands when
        the segment is persisted — persisted before broadcast, so turn-level
        attribution, including what a tool handler wrote mid-loop, lands in
        the stored row and in the stream_end frame without any post-hoc
        rewrite.

        ``cancelled`` marks a segment cut short by an interrupted turn, so a
        reader can tell a finished answer from one the user stopped.
        """
        # Joined even with nothing to write: the caller's last flush is what
        # lets a write cut off by a stop land before the turn returns.
        await self._settle()
        if not self._accumulated:
            return None
        body = "".join(self._accumulated)
        self._accumulated.clear()
        metadata = dict(self._sr.response_metadata or {})
        # Read after the loop's end: this is the turn's final text
        carries_record = self._turn_record is not None
        if self._turn_record is not None:
            metadata.update(self._turn_record)
        if cancelled:
            metadata["cancelled"] = True
        event = self._build(EventType.MESSAGE, TextContent(body=body), metadata=metadata)
        # The streaming channel already rendered this text chunk by chunk —
        # only the others get it as an event.
        row = await self._write(event, exclude=set(self._streamed_to))
        if carries_record and row is not None:
            self._record_owed = False
        return row

    def started(self, tool_id: str) -> bool:
        """Whether this call's start row was handed to the writer."""
        return tool_id in self._started

    async def take(self, marker: Any) -> list[RoomEvent]:
        """Handle a stream marker; the rows it committed, in order.

        A call's start ends the text before it, which is its own segment. The
        loop's end is recorded, and written with the turn's last message.
        """
        rows: list[RoomEvent | None] = []
        if isinstance(marker, ToolCallStartMarker):
            rows = [await self.flush_text(), await self.tool_start(marker)]
        elif isinstance(marker, ToolCallEndMarker):
            rows = [await self.tool_end(marker)]
        elif isinstance(marker, LoopEndMarker):
            self.end_turn(marker)
        return [row for row in rows if row is not None]

    async def tool_start(self, marker: ToolCallStartMarker) -> RoomEvent | None:
        return await self._write(self._start_row(marker), exclude=set(self._streamed_to))

    async def tool_end(self, marker: ToolCallEndMarker) -> RoomEvent | None:
        # Excluded like a text segment, and for the same reason: the channel
        # consuming the stream is handed every persisted event inline, so
        # laning it there too sent the same event id twice.
        return await self._write(self._end_row(marker), exclude=set(self._streamed_to))

    async def close_calls(
        self, closed: list[tuple[ToolCallStartMarker, ToolCallEndMarker]]
    ) -> None:
        """Write the end of each call a stop or an abandon closed.

        The stream no longer carries rows, so each goes to every channel, the
        one that streamed included: it saw the start inline and would keep
        the call running. A call whose start row was cut off gets it first
        (RFC §12.2 step 13s).
        """
        for start, end in closed:
            if not self.started(start.tool_id):
                await self._write(self._start_row(start), exclude=None)
            await self._write(self._end_row(end), exclude=None)

    def _start_row(self, marker: ToolCallStartMarker) -> RoomEvent:
        self._started.add(marker.tool_id)
        return self._build(
            EventType.TOOL_CALL_START,
            ToolCallContent(
                tool_name=marker.tool_name,
                tool_id=marker.tool_id,
                arguments=marker.arguments,
                status="pending",
            ),
        )

    def _end_row(self, marker: ToolCallEndMarker) -> RoomEvent:
        result, structured = tool_event_payload(marker.result, marker.structured_content)
        return self._build(
            EventType.TOOL_CALL_END,
            ToolCallContent(
                tool_name=marker.tool_name,
                tool_id=marker.tool_id,
                arguments=marker.arguments,
                result=result,
                status=marker.status,
                duration_ms=marker.duration_ms,
                error=marker.error,
                structured_content=structured,
            ),
        )

    # -- how any of them is written ----------------------------------------

    def _build(
        self,
        event_type: EventType,
        content: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RoomEvent:
        return RoomEvent(
            room_id=self._room_id,
            source=EventSource(
                channel_id=self._sr.source_channel_id,
                channel_type=self._sr.source_channel_type,
            ),
            type=event_type,
            content=content,
            status=EventStatus.DELIVERED,
            chain_depth=self._chain_depth,
            visibility=self._visibility,
            correlation_id=self._correlation_id,
            parent_event_id=self._parent_event_id,
            metadata=metadata or {},
        )

    async def _settle(self) -> None:
        """Wait for every row whose write outlived the read that started it.

        ``asyncio.wait`` rather than ``gather``: a waiter cancelled here must
        not cancel the writes it was waiting on.
        """
        while self._writing:
            await asyncio.wait(set(self._writing))

    async def _write(self, event: RoomEvent, *, exclude: set[str] | None) -> RoomEvent | None:
        """Gate and commit a row, even when the read that produced it is cancelled.

        A barge-in cancels the stream's in-flight read wherever it stands,
        possibly halfway through a commit. The row's text has already left
        the buffer, so a cancelled commit would lose it: the write finishes
        on its own. Every row first waits for the writes before it, so rows
        commit in the order the stream produced them.
        """
        await self._settle()
        task = asyncio.ensure_future(self._write_now(event, exclude=exclude))
        self._writing.add(task)
        task.add_done_callback(self._writing.discard)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Nobody awaits the write any more: its failure is logged here.
            task.add_done_callback(self._log_orphan_failure)
            raise

    def _log_orphan_failure(self, task: asyncio.Task[RoomEvent | None]) -> None:
        if not task.cancelled() and (exc := task.exception()) is not None:
            logger.error(
                "Streamed row write failed after its read was cancelled (room %s)",
                self._room_id,
                exc_info=exc,
            )

    async def _write_now(self, event: RoomEvent, *, exclude: set[str] | None) -> RoomEvent | None:
        gated = await self._gate(event)
        if gated is None:
            return None
        event, hook_result = gated
        return await self._lane(event, exclude=exclude, hook_result=hook_result)

    async def _gate(self, event: RoomEvent) -> tuple[RoomEvent, SyncPipelineResult] | None:
        """Run the BEFORE_BROADCAST sync hooks on a row before it commits.

        Mirrors the locked path. The live chunks already piped to streaming
        channels are outside a hook's reach by construction; this lands any
        hook modification (e.g. PII de-anonymisation, a display label on a
        tool call) on the persisted row and on the delivery to the
        non-streaming channels.

        ``None`` when a hook refused the row. A refusal is not a drop: it
        goes through the same block handler as every other one (RFC §10.1
        step 10) — committed with status BLOCKED as the audit record,
        announced as ``event_blocked`` — and what the hook decided still
        stands, its tasks and observations persisted and its injected events
        laned.
        """
        room_id = self._room_id
        sync_result = await self._kit._hook_engine.run_sync_hooks(
            room_id, HookTrigger.BEFORE_BROADCAST, event, self._context
        )
        if sync_result.hook_errors:
            logger.warning(
                "BEFORE_BROADCAST hook error on streamed %s (room %s): %s",
                event.type.value,
                room_id,
                sync_result.hook_errors,
            )
        if not sync_result.allowed:
            blocked = await self._kit._handle_block(
                room_id=room_id,
                event=event,
                reason=sync_result.reason,
                blocked_by=sync_result.blocked_by,
                injected_events=sync_result.injected_events,
                context=self._context,
                cascade=self._cascade,
            )
            await self._kit._persist_side_effects(
                room_id, sync_result.tasks, sync_result.observations, blocked, self._context
            )
            logger.info(
                "Streamed %s blocked by BEFORE_BROADCAST hook (room %s): %s",
                event.type.value,
                room_id,
                sync_result.reason,
            )
            return None
        if isinstance(sync_result.event, RoomEvent):
            event = sync_result.event
        return event, sync_result

    async def _lane(
        self,
        event: RoomEvent,
        *,
        exclude: set[str] | None,
        hook_result: SyncPipelineResult,
    ) -> RoomEvent | None:
        """Commit a row and queue its delivery on the room's lane.

        Deliberately not awaited to completion: this runs inside the
        streaming channel's ``deliver_stream``, and blocking the generator on
        a transport round trip would stall the stream. The cascade collects
        every row's unit instead, and the caller waits on it once the stream
        is done.
        """
        stored = await self._kit._commit_and_deliver(
            self._room_id,
            event,
            self._plan_source,
            exclude_delivery=exclude,
            cascade=self._cascade,
            hook_result=hook_result,
        )
        if stored is not None:
            self.persisted.append(stored)
            if self._response_events is not None:
                self._response_events.append(stored)
        return stored
