"""InboundStreamingMixin — streaming response handling outside the room lock."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.core.lanes import DeliveryCascade
from roomkit.core.mixins._streaming_segments import SegmentWriter
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.core.mixins.lane_execution import DeliverySource
from roomkit.core.visibility import visibility_allows
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelDirection,
)
from roomkit.models.event import RoomEvent
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.streaming import ThinkingDeltaMarker, ToolCallEndMarker, ToolCallStartMarker
from roomkit.providers.ai.base import ProviderError
from roomkit.providers.utils import _aclose_stream

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.event_router import EventRouter, StreamingResponse
    from roomkit.core.hooks import HookEngine
    from roomkit.models.context import RoomContext
    from roomkit.models.hook import InjectedEvent
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")


@dataclass
class _StreamingResult:
    """Result of handling a streaming response.

    ``error`` is the exception raised while consuming the response stream
    (provider/transport failure), captured so the inbound pipeline can surface
    it to a headless caller. ``None`` when the stream completed.
    """

    events: list[RoomEvent] = field(default_factory=list)
    error: Exception | None = None


@runtime_checkable
class InboundStreamingHost(Protocol):
    """Contract: capabilities a host class must provide for InboundStreamingMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation store for event persistence.
        _channels: Channel registry.
        _hook_engine: Hook engine for AFTER_BROADCAST / ON_ERROR hooks.
        _max_chain_depth: Maximum chain depth to prevent infinite loops.

    Methods provided by the host class (RoomKit):
        _get_router: Lazily create / return the ``EventRouter`` for broadcast.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _hook_engine: HookEngine
    _max_chain_depth: int

    def _get_router(self) -> EventRouter: ...

    async def _lane_injected_events(
        self,
        injected_events: list[InjectedEvent],
        room_id: str,
        context: RoomContext,
        cascade: DeliveryCascade,
    ) -> None: ...


class InboundStreamingMixin(HelpersMixin):
    """Streaming response handling extracted from the inbound pipeline.

    These methods run outside the room lock so that streaming delivery
    (e.g. TTS playback) does not block other ``process_inbound`` calls.

    Host contract: :class:`InboundStreamingHost`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _hook_engine: HookEngine
    _max_chain_depth: int

    # Cross-mixin method — attribute annotation avoids MRO shadowing
    _commit_and_deliver: Any  # LaneExecutionMixin
    _lane_injected_events: Any  # LaneExecutionMixin
    _handle_block: Any  # InboundLockedMixin

    # Stub for cross-mixin call — implemented by RoomKit._get_router().
    def _get_router(self) -> EventRouter: ...

    async def _handle_streaming_response(
        self,
        router: EventRouter,
        sr: StreamingResponse,
        room_id: str,
        context: RoomContext,
        *,
        response_events: list[RoomEvent] | None = None,
        cascade: DeliveryCascade | None = None,
    ) -> _StreamingResult | None:
        """Consume a streaming response, pipe to streaming channels, store segments."""
        from roomkit.models.event import EventSource, TextContent

        response_vis = sr.trigger_event.response_visibility
        streaming_targets = self._find_streaming_targets(router, sr, context)

        logger.debug(
            "Streaming targets for room %s: %d found",
            room_id,
            len(streaming_targets),
        )

        # One cascade for the whole response: each segment's delivery is
        # enqueued without waiting (blocking the generator on an SMS round
        # trip would stall the stream), and the run is awaited once, after
        # the stream, by the caller.
        if cascade is None:
            cascade = DeliveryCascade(room_id, reentry_budget=self._max_chain_depth * 10)
        # Only the first target streams (V1, below); any other
        # streaming-capable channel is an ordinary recipient.
        streamed_to: set[str] = (
            {streaming_targets[0][1].channel_id} if streaming_targets else set()
        )
        correlation_id = uuid4().hex
        chain_depth = sr.trigger_event.chain_depth + 1
        visibility = response_vis or "all"
        # Inherit the trigger's thread root so the AI reply lands in the same
        # thread (already normalised to a root by the locked pipeline). None
        # when the trigger is top-level — the reply stays top-level too.
        parent_event_id = sr.trigger_event.parent_event_id

        # Planning inputs for the whole run, resolved once. Every segment has
        # the same sender and the same delivery set, so re-resolving per
        # segment would buy nothing and cost a room lock plus a context read
        # each time — on a tool-heavy turn, tens of them on the hot path the
        # delivery lanes exist to keep clear. The binding is already in the
        # context the caller built; the batch broadcast this replaced planned
        # off that same snapshot.
        source_binding = next(
            (b for b in context.bindings if b.channel_id == sr.source_channel_id), None
        )
        plan_source: DeliverySource | str = (
            DeliverySource(binding=source_binding, context=context)
            if source_binding is not None
            else sr.source_channel_id
        )

        writer = SegmentWriter(
            self,
            sr,
            room_id=room_id,
            context=context,
            cascade=cascade,
            plan_source=plan_source,
            chain_depth=chain_depth,
            visibility=visibility,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            streamed_to=streamed_to,
            response_events=response_events,
        )

        # Whether the transport read the response to its end. A transport can
        # hand back early (every voice session barged in, RFC §12.2 step 13s):
        # the final flush below then never runs, and the response is closed
        # and stored cancelled after deliver_stream() returns.
        exhausted = False

        # Generator that yields text deltas and persisted events.
        # Text deltas drive the streaming bubble; RoomEvents are delivered
        # as regular events interleaved between stream chunks.
        async def segment_stream() -> Any:
            """Yield str for text deltas, RoomEvent for persisted segments.

            Thinking markers pass straight through to the channel — they
            carry transient display info only and are not persisted as
            RoomEvents (the realtime bus still publishes a buffered
            ``THINKING_END`` for out-of-band observers).
            """
            nonlocal exhausted
            async for delta in sr.stream:
                if isinstance(delta, str):
                    writer.add_text(delta)
                    yield delta
                elif isinstance(delta, ThinkingDeltaMarker):
                    yield delta
                elif isinstance(delta, ToolCallStartMarker):
                    # The text before the tool call is its own segment.
                    for row in (await writer.flush_text(), await writer.tool_start(delta)):
                        if row is not None:
                            yield row
                elif isinstance(delta, ToolCallEndMarker):
                    row = await writer.tool_end(delta)
                    if row is not None:
                        yield row

            exhausted = True
            row = await writer.flush_text()
            if row is not None:
                yield row

        stream_error: Exception | None = None
        if streaming_targets:
            channel, binding = streaming_targets[0]  # V1: single target
            placeholder = RoomEvent(
                room_id=room_id,
                source=EventSource(
                    channel_id=sr.source_channel_id,
                    channel_type=sr.source_channel_type,
                ),
                content=TextContent(body=""),
                chain_depth=chain_depth,
                visibility=visibility,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
            segments = segment_stream()
            try:
                await channel.deliver_stream(segments, placeholder, binding, context)
                if not exhausted:
                    # The transport stopped reading: the user cut the response
                    # off. Close the generation first, so no token is produced
                    # and no tool call starts past this point, then keep what
                    # was already produced, marked like any interrupted turn.
                    await segments.aclose()
                    await _aclose_stream(sr.stream)
                    await writer.flush_text(cancelled=True)
            except asyncio.CancelledError:
                # A turn interrupted on purpose (the console's Esc). What was
                # already streamed is on the user's screen, so the timeline
                # MUST hold it too: dropping it would leave the room
                # disagreeing with what the human read, and the agent's next
                # context missing what it already said. Not an error — nobody
                # failed — so ON_ERROR stays silent and the cancellation
                # propagates untouched.
                await writer.flush_text(cancelled=True)
                raise
            except Exception as exc:
                stream_error = exc
                self._log_stream_failure(
                    exc, f"streaming delivery to {binding.channel_id}", room_id
                )
                # Persist any text accumulated before the error. The stream is
                # gone, so this text never reached its channels — it goes out
                # as an ordinary event, to everyone.
                writer.stream_lost()
                await writer.flush_text()
                await self._fire_error_hook(
                    room_id,
                    context,
                    EventSource(
                        channel_id=sr.source_channel_id,
                        channel_type=sr.source_channel_type,
                    ),
                    error=str(exc),
                    error_type=type(exc).__name__,
                    error_category="streaming",
                    chain_depth=chain_depth,
                    visibility=visibility,
                    correlation_id=correlation_id,
                    parent_event_id=parent_event_id,
                )
        else:
            # No streaming targets (e.g. a PII-locked / edge agent whose stream
            # send fn was withheld, or a headless one-shot call whose only
            # transport is also the source) — still consume the stream to drive
            # persistence via markers. Under the SAME error contract as the
            # streaming branch above: a failure (context overflow, provider
            # error) must fire ON_ERROR so the error reaches the ON_ERROR hooks
            # (which classify + surface it) AND be returned to the caller via
            # ``_StreamingResult.error``, instead of vanishing with no card.
            try:
                async for _ in segment_stream():
                    pass
            except asyncio.CancelledError:
                await writer.flush_text(cancelled=True)
                raise
            except Exception as exc:
                stream_error = exc
                self._log_stream_failure(
                    exc, "stream consumption (no targets)", room_id, headless=True
                )
                await writer.flush_text()
                await self._fire_error_hook(
                    room_id,
                    context,
                    EventSource(
                        channel_id=sr.source_channel_id,
                        channel_type=sr.source_channel_type,
                    ),
                    error=str(exc),
                    error_type=type(exc).__name__,
                    error_category="streaming",
                    chain_depth=chain_depth,
                    visibility=visibility,
                    correlation_id=correlation_id,
                    parent_event_id=parent_event_id,
                )

        # Every segment's delivery set, awaited once now that the stream is
        # done — the run's completion is what the caller's turn waits on.
        await cascade.wait()

        if not writer.persisted and stream_error is None:
            return None

        return _StreamingResult(events=writer.persisted, error=stream_error)

    @staticmethod
    def _log_stream_failure(
        exc: Exception, what: str, room_id: str, *, headless: bool = False
    ) -> None:
        """Log a streaming-response failure at the right verbosity.

        A ``ProviderError`` (backend unreachable, 5xx, timeout, context
        overflow) is a transient/expected condition, not a code defect — no
        traceback, and the error is also returned to the caller and delivered to
        ``ON_ERROR`` hooks. When there is no streaming target (``headless`` — a
        one-shot programmatic caller that owns its own logging), a framework
        WARNING would just duplicate the caller's line, so it drops to DEBUG;
        with a streaming target the framework WARNING is the operational record.
        Any other exception is unexpected and keeps its full traceback.
        """
        if isinstance(exc, ProviderError):
            level = logging.DEBUG if headless else logging.WARNING
            logger.log(level, "%s failed for room %s: %s", what, room_id, exc)
        else:
            logger.exception("%s failed for room %s", what, room_id)

    def _find_streaming_targets(
        self,
        router: Any,
        sr: Any,
        context: RoomContext,
    ) -> list[Any]:
        """Find transport channels that support streaming delivery."""
        response_vis = sr.trigger_event.response_visibility
        targets: list[Any] = []
        for binding in context.bindings:
            if binding.category != ChannelCategory.TRANSPORT:
                continue
            if binding.channel_id == sr.source_channel_id:
                continue
            if binding.access in (Access.WRITE_ONLY, Access.NONE):
                continue
            if binding.direction == ChannelDirection.OUTBOUND:
                continue
            if response_vis is not None and not visibility_allows(response_vis, binding):
                continue
            channel = router.get_channel(binding.channel_id)
            # Asked per room: a channel can hold streaming clients for one room
            # and none for another, and only the room being delivered counts.
            supports = (
                channel.supports_streaming_delivery_for(binding.room_id) if channel else False
            )
            if channel and supports:
                targets.append((channel, binding))
        return targets

    async def _process_streaming_responses(
        self,
        pending_streams: list[Any],
        room_id: str,
        *,
        response_events: list[RoomEvent] | None = None,
        cascade: DeliveryCascade | None = None,
    ) -> tuple[Exception | None, ResponseMetadata]:
        """Handle streaming responses outside the room lock.

        Streaming delivery (TTS playback) can take seconds. Running it outside
        the lock allows other process_inbound calls to proceed concurrently,
        preventing continuous STT echo from being queued behind the lock.

        Each segment commits and reaches the non-streaming channels through
        the room's delivery lane as it is produced (RFC §10.2 — the lane is
        the room's single ordering authority, and its executor fires each
        segment's AFTER_BROADCAST once that segment's delivery set has run,
        step 16). Broadcasting the run in one batch after the stream is what
        used to let the cursor run ahead of the deliveries.

        Returns the first response-stream failure encountered (so the inbound
        pipeline can surface it to a headless caller), or ``None`` when every
        stream completed, plus the turn's response-metadata record.

        The record is handed back rather than left to be read off a persisted
        segment: a turn that ends on a tool call persists no segment after it,
        and one that ends before writing any text persists none at all, so the
        room is not a place where "how did that turn end" can always be asked.
        """
        router = self._get_router()
        context = await self._build_context(room_id)

        first_error: Exception | None = None
        record = ResponseMetadata()
        try:
            for sr in pending_streams:
                sr_result = await self._handle_streaming_response(
                    router, sr, room_id, context, response_events=response_events, cascade=cascade
                )
                if sr_result and sr_result.error and first_error is None:
                    first_error = sr_result.error
                # Several streams answer one inbound only when several channels
                # replied; each writes under its own key, so merging keeps them
                # all rather than letting the last one win.
                record.update(sr.response_metadata or {})
        finally:
            # A transport can stop reading between two yields (or fail while
            # rendering one). Async-for alone does not close its generator;
            # finalizers must run before the delivery handle reports cleanup.
            for sr in pending_streams:
                await _aclose_stream(sr.stream)

        return first_error, record
