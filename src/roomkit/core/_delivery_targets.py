"""Destination validation and outcome reporting for proactive delivery."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import DeliveryError, DeliveryOutcome, InboundMessage, InboundResult
from roomkit.models.enums import Access, ChannelCategory, ChannelType, EventStatus, RoomStatus
from roomkit.models.event import TextContent
from roomkit.voice.base import VoiceSessionState

if TYPE_CHECKING:
    from roomkit.core.delivery import DeliveryContext


def unavailable(reason: str, targets: list[str] | None = None) -> DeliveryOutcome:
    """A destination may become available on a later attempt."""
    return DeliveryOutcome(
        status="unavailable",
        reason=reason,
        unavailable_targets=targets or [],
        error=DeliveryError(code=reason, message=reason),
    )


async def prepare_delivery(
    ctx: DeliveryContext,
    channel_id: str | None = None,
) -> tuple[str | None, DeliveryOutcome | None]:
    """Resolve and validate the channel; pin realtime sessions before waiting."""
    channel_id = channel_id or await ctx.resolve_channel_id()
    if channel_id is None:
        return None, unavailable("no_transport")
    channel = ctx.kit.get_channel(channel_id)
    bindings = await ctx.kit.store.list_bindings(ctx.room_id)
    binding = next((b for b in bindings if b.channel_id == channel_id), None)
    if channel is None or binding is None:
        return None, unavailable("channel_unavailable", [channel_id])
    if channel.category == ChannelCategory.INTELLIGENCE:
        transport_id = await ctx.find_transport_channel_id()
        if transport_id is None:
            return None, unavailable("no_transport")
        return await prepare_delivery(ctx, transport_id)
    if channel.channel_type != ChannelType.REALTIME_VOICE or ctx.addressed_to is not None:
        if ctx.session_id is not None:
            return None, DeliveryOutcome(status="blocked", reason="session_requires_realtime")
        return channel_id, None
    room = await ctx.kit.store.get_room(ctx.room_id)
    if room is None:
        return None, unavailable("room_unavailable")
    if room.status in (RoomStatus.CLOSED, RoomStatus.ARCHIVED):
        return None, DeliveryOutcome(status="blocked", reason="room_closed")
    if binding.access in (Access.WRITE_ONLY, Access.NONE):
        return None, DeliveryOutcome(status="blocked", reason="channel_cannot_read")
    if ctx._voice_channel is not None and ctx._voice_channel is not channel:
        return None, unavailable("voice_channel_replaced", [channel_id])
    if ctx._voice_sessions is None:
        sessions = _active_sessions(channel, ctx.room_id)
        if ctx.session_id is not None:
            sessions = [s for s in sessions if s.id == ctx.session_id]
        elif ctx.channel_id is not None and len(sessions) > 1:
            return None, unavailable("ambiguous_voice_session", [channel_id])
        if not sessions:
            return None, unavailable("voice_session_unavailable", [ctx.session_id or channel_id])
        ctx._voice_channel = channel
        ctx._voice_sessions = sessions
    return channel_id, None


def _active_sessions(channel: Any, room_id: str) -> list[Any]:
    return [s for s in channel.get_room_sessions(room_id) if s.state != VoiceSessionState.ENDED]


async def deliver_to_channel(ctx: DeliveryContext, channel_id: str) -> DeliveryOutcome:
    """Use the existing inbound pipeline or the selected realtime provider."""
    resolved_id, refusal = await prepare_delivery(ctx, channel_id)
    if refusal is not None:
        return refusal
    assert resolved_id is not None
    channel_id = resolved_id
    channel = ctx.kit.get_channel(channel_id)
    if channel is None:
        return unavailable("channel_unavailable", [channel_id])
    if channel.channel_type == ChannelType.REALTIME_VOICE and ctx.addressed_to is None:
        return await deliver_to_realtime_voice(channel, ctx)
    result = await ctx.kit.process_inbound(
        InboundMessage(
            channel_id=channel_id,
            sender_id="system",
            content=TextContent(body=ctx.content),
            metadata=ctx.metadata or {},
            addressed_to=ctx.addressed_to,
            idempotency_key=ctx.idempotency_key,
        ),
        room_id=ctx.room_id,
        defer_delivery=True,
    )
    if not isinstance(result, InboundResult):
        return DeliveryOutcome(status="unknown", reason="inbound_outcome_unknown")
    if result.delivery is not None and ctx._wait_for_turn:
        try:
            await result.delivery.wait()
        except asyncio.CancelledError:
            await result.delivery.cancel()
            raise
    return await _text_outcome(ctx, result)


async def _text_outcome(ctx: DeliveryContext, result: InboundResult) -> DeliveryOutcome:
    event = result.event
    outcome = DeliveryOutcome(
        status="sent",
        inbound=result,
        event_id=event.id if event is not None else None,
        duplicate=result.duplicate,
        turn_complete=bool(result.delivery and result.delivery.done and not result.duplicate),
    )
    if result.blocked or (event is not None and event.status == EventStatus.BLOCKED):
        return outcome.model_copy(
            update={"status": "blocked", "reason": result.reason or "inbound_blocked"}
        )
    if result.error is not None or result.cancellation_reason is not None:
        error = result.error
        return outcome.model_copy(
            update={
                "status": "failed",
                "reason": result.cancellation_reason or "inbound_failed",
                "error": DeliveryError(
                    code=type(error).__name__ if error else "turn_cancelled",
                    message=str(error) if error else str(result.cancellation_reason),
                    retryable=event is None,
                ),
            }
        )
    failed = next((r for r in result.delivery_results.values() if r.status == "failed"), None)
    if failed is not None:
        error = failed.error or DeliveryError(
            code="delivery_failed", message="Channel delivery failed"
        )
        return outcome.model_copy(
            update={
                "status": "failed",
                "reason": "channel_delivery_failed",
                "error": error.model_copy(update={"retryable": event is None}),
            }
        )
    if event is None:
        return outcome.model_copy(update={"status": "unknown", "reason": "no_publication_result"})
    # Replayed calls identify the original publication, not a new attempt at
    # the address supplied on this call. Its turn's completion is unknown.
    if result.duplicate:
        outcome.reason = "duplicate_publication"
    addresses = event.addressed_to
    if addresses:
        context = await ctx.kit._build_context(ctx.room_id)  # noqa: SLF001
        source = next(
            (b for b in context.bindings if b.channel_id == event.source.channel_id), None
        )
        targets = (
            ctx.kit._get_router().plan(event, source, context).targets  # noqa: SLF001
            if source is not None
            else []
        )
        eligible = {
            b.channel_id
            for b in targets
            if b.category == ChannelCategory.INTELLIGENCE
            and ctx.kit.get_channel(b.channel_id) is not None
        }
        missing = [target for target in addresses if target not in eligible]
        if missing:
            return outcome.model_copy(
                update={
                    "status": "unavailable",
                    "reason": "addressed_targets_unavailable",
                    "unavailable_targets": missing,
                    "error": DeliveryError(
                        code="addressed_targets_unavailable",
                        message="The event was published but an addressed target was unavailable",
                        retryable=False,
                    ),
                }
            )
    return outcome


async def deliver_to_realtime_voice(channel: Any, ctx: DeliveryContext) -> DeliveryOutcome:
    """Inject only pinned sessions and retain partial progress on failure.

    Provider acceptance and store publication are not one transaction. A
    retry may inject again, including when an idempotency key was supplied.
    """
    sessions = ctx._voice_sessions
    if sessions is None:
        sessions = _active_sessions(channel, ctx.room_id)
    if not sessions:
        return unavailable("voice_session_unavailable")
    outcome = DeliveryOutcome(
        status="sent",
        reason="voice_not_deduplicated" if ctx.idempotency_key is not None else None,
    )
    for session in sessions:
        if not any(session is active for active in _active_sessions(channel, ctx.room_id)):
            return outcome.model_copy(
                update={
                    "status": "unavailable",
                    "reason": "voice_session_replaced",
                    "error": DeliveryError(
                        code="voice_session_replaced", message="Pinned session ended or changed"
                    ),
                }
            )
        try:
            bindings = await ctx.kit.store.list_bindings(ctx.room_id)
            binding = next((b for b in bindings if b.channel_id == channel.channel_id), None)
            if binding is None or ctx.kit.get_channel(channel.channel_id) is not channel:
                return unavailable("channel_unavailable", [channel.channel_id])
            if binding.access in (Access.WRITE_ONLY, Access.NONE):
                return DeliveryOutcome(status="blocked", reason="channel_cannot_read")
            silent = binding.muted or binding.output_muted or not binding.can_write
            if silent:
                await channel.inject_text(session, ctx.content, silent=True)
            else:
                await channel.inject_text(session, ctx.content)
        except Exception as exc:
            return outcome.model_copy(
                update={
                    "status": "failed",
                    "reason": "voice_injection_failed",
                    "error": DeliveryError(code=type(exc).__name__, message=str(exc)),
                }
            )
        if not any(session is active for active in _active_sessions(channel, ctx.room_id)):
            return outcome.model_copy(
                update={
                    "status": "unavailable",
                    "reason": "voice_session_replaced",
                    "error": DeliveryError(
                        code="voice_session_replaced", message="Session changed during injection"
                    ),
                }
            )
        outcome.session_ids.append(session.id)
    return outcome
