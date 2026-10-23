"""InjectedEventsMixin — the events a hook injects (RFC §9.4).

Each one is committed as a pass of its cause's cascade (§8.3, §10.1 step 14):
it takes the room lock for its own commit, runs the ``BEFORE_BROADCAST``
chain with every hook under the lock, and counts against the cascade's
reentry budget. It keeps its cause's chain depth, names its targets as its
visibility, carries no idempotency key, and takes no source check: it is the
hook's, not a channel's. It is delivered to its transport targets only,
under the rules of any delivery, never to its own source, and asks no agent.
An injected INSTRUCTION is dropped with a warning before any pass: only the
application issues instructions.

A pass started inside its cause's locked section takes the lock reentrantly,
so it commits right after its cause with nothing between them: a stricter
order than the RFC's, which runs the queued passes once the cause has
released the lock.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.core._instructions import without_fingerprint
from roomkit.core.lanes import DeliveryCascade, DeliveryPlan
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.core.mixins.inbound_locked import _Blocked, _Ready, mutation_plan, target_update
from roomkit.core.mixins.lane_execution import provisional
from roomkit.core.visibility import intersect_visibility, visibility_allows
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    EventStatus,
    EventType,
    HookTrigger,
    Visibility,
)
from roomkit.models.event import RoomEvent, with_pipeline_id

if TYPE_CHECKING:
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.context import RoomContext
    from roomkit.models.hook import InjectedEvent

logger = logging.getLogger("roomkit.framework")


def _as_injected(event: RoomEvent, cause: RoomEvent) -> RoomEvent:
    """*event* as its pass commits it (RFC §9.4): in its cause's room, at
    its cause's chain depth, with no idempotency key (§13.4) and no
    instruction fingerprint, the pipeline's on a response alone (§5.2),
    under an id the pipeline gives (§8.1)."""
    return with_pipeline_id(
        event.model_copy(
            update={
                "room_id": cause.room_id,
                "chain_depth": cause.chain_depth,
                "idempotency_key": None,
                "metadata": without_fingerprint(event.metadata),
            }
        )
    )


def _without_instructions(
    injected_events: list[InjectedEvent], cause: RoomEvent
) -> list[InjectedEvent]:
    """*injected_events* without the instructions among them (RFC §9.4,
    §10.1.1): only the application issues an instruction, which a hook sends
    with ``send_event()``. Each one dropped is logged, before any pass: no
    hook runs on it, and nothing of it is stored or delivered."""
    kept: list[InjectedEvent] = []
    for injected in injected_events:
        if injected.event.type != EventType.INSTRUCTION:
            kept.append(injected)
            continue
        logger.warning(
            "A hook injected an INSTRUCTION while deciding on event %s: dropped, since only "
            "the application issues instructions (send_event); the rest of its result stands",
            cause.id,
            extra={"room_id": cause.room_id, "event_id": cause.id},
        )
    return kept


def _targets_scope(injected: InjectedEvent, context: RoomContext) -> str:
    """The visibility an injected event is committed with: the targets the
    room binds, whatever visibility it came with, and ``none`` when it is
    stored only (RFC §9.4)."""
    bound = {binding.channel_id for binding in context.bindings}
    named = sorted(bound.intersection(injected.target_channel_ids or ()))
    return ",".join(named) if named else Visibility.NONE


class InjectedEventsMixin(HelpersMixin):
    """The passes of the events hooks inject, the identity challenge's included.

    Host contract: the lane execution and the locked pipeline.
    """

    _lock_manager: RoomLockManager
    _max_chain_depth: int

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _commit_to_lane: Any  # LaneExecutionMixin
    _finish_cascade: Any  # LaneExecutionMixin
    _open_pass: Any  # LaneExecutionMixin
    _store_past_reentry_cap: Any  # LaneExecutionMixin
    _check_change: Any  # InboundLockedMixin
    _gate_commit: Any  # InboundLockedMixin
    _store_blocked: Any  # InboundLockedMixin

    async def _lane_injected_events(
        self,
        injected_events: list[InjectedEvent],
        room_id: str,
        cascade: DeliveryCascade,
        *,
        cause: RoomEvent,
    ) -> None:
        """Commit each event *cause*'s hooks injected as a pass of
        *cascade*, in order (RFC §9.4), whichever way *cause* ended; an
        instruction among them is dropped before any pass."""
        for injected in _without_instructions(injected_events, cause):
            await self._run_injected_pass(room_id, injected, cascade, cause=cause)

    async def _inject_off_lock(
        self, injected_events: list[InjectedEvent], room_id: str, *, cause: RoomEvent
    ) -> None:
        """Commit and deliver events injected outside any cascade (an
        identity challenge, before the room lock), waiting for their
        delivery."""
        cascade = DeliveryCascade(room_id, reentry_budget=self._max_chain_depth * 10)
        await self._lane_injected_events(injected_events, room_id, cascade, cause=cause)
        await self._finish_cascade(cascade, room_id)

    async def _run_injected_pass(
        self,
        room_id: str,
        injected: InjectedEvent,
        cascade: DeliveryCascade,
        *,
        cause: RoomEvent,
    ) -> None:
        """One injected event's own commit pass, under the room lock."""
        async with self._lock_manager.locked(room_id):
            event = _as_injected(injected.event, cause)
            room_context = await self._open_pass(room_id, event)
            if room_context is None:
                return
            event = event.model_copy(update={"visibility": _targets_scope(injected, room_context)})
            event, context = provisional(event, room_context)
            if not cascade.consume_reentry_budget():
                # Past the budget it is not a pass: its hooks never run (§8.3).
                await self._store_past_reentry_cap(room_id, event)
                return
            # An edit or a delete it carries is the hook's, a SYSTEM change
            # only (RFC §10.3 rule 6); a refused one is logged, nothing stored.
            check = await self._check_change(room_id, event, by_hook=True)
            if check.refusal is not None:
                return
            decision = await self._gate_commit(room_id, event, context, source_check=False)
            if isinstance(decision, _Blocked):
                await self._store_blocked(room_id, decision, cascade)
                return
            decision.edit_delete_target = check.target
            await self._commit_injected(room_id, decision, injected, event.visibility, cascade)

    async def _commit_injected(
        self,
        room_id: str,
        ready: _Ready,
        injected: InjectedEvent,
        scope: str,
        cascade: DeliveryCascade,
    ) -> None:
        """Commit an injected event its hooks let through and lane its
        delivery, under the lock its pass holds; then its hooks' side
        effects and the events they injected in turn."""
        context = ready.context
        categories = {binding.channel_id: binding.category for binding in context.bindings}
        event = ready.event.model_copy(
            update={
                "status": EventStatus.DELIVERED,
                # A hook may narrow the scope of its targets, never widen it.
                "visibility": intersect_visibility(ready.event.visibility, scope, categories.get),
            }
        )
        targets = await self._injected_targets(room_id, event, injected)
        # An edit's or a delete's target, written in the event's own commit
        # (RFC §10.3, §14.3).
        target, mutation_hook = target_update(event, ready.edit_delete_target)

        def factory(committed: RoomEvent) -> DeliveryPlan | None:
            return _injected_plan(committed, ready, targets, mutation_hook)

        stored = await self._commit_to_lane(
            room_id, event, cascade, factory, policy_aware=False, target=target
        )
        committed = stored or event
        sync_result = ready.sync_result
        await self._persist_side_effects(
            room_id, sync_result.tasks, sync_result.observations, committed, context
        )
        if sync_result.injected_events:
            await self._lane_injected_events(
                sync_result.injected_events, room_id, cascade, cause=committed
            )

    async def _injected_targets(
        self, room_id: str, event: RoomEvent, injected: InjectedEvent
    ) -> list[ChannelBinding]:
        """The transports *event* is delivered to: its targets as the room
        binds them now (a hook of its pass may have changed a binding),
        under the rules of any delivery, never its own source (RFC §9.4,
        §7.5 rule 5). An agent among them is not called: it reads the event
        in its history."""
        targets: list[ChannelBinding] = []
        for target_id in dict.fromkeys(injected.target_channel_ids or ()):
            if target_id == event.source.channel_id:
                continue
            binding = await self._store.get_binding(room_id, target_id)
            if (
                binding is not None
                and binding.category == ChannelCategory.TRANSPORT
                and binding.access not in (Access.WRITE_ONLY, Access.NONE)
                and visibility_allows(event.visibility, binding)
            ):
                targets.append(binding)
        return targets


def _injected_plan(
    committed: RoomEvent,
    ready: _Ready,
    targets: list[ChannelBinding],
    mutation_hook: tuple[HookTrigger, RoomEvent] | None,
) -> DeliveryPlan | None:
    """The delivery plan of an injected event: its transport targets, no
    reentry, no ``AFTER_BROADCAST``, and the mutation trigger of an edit or
    a delete (RFC §10.3); ``None`` when it is stored only and changes
    nothing."""
    if not targets:
        return mutation_plan(committed, ready.context, mutation_hook, injected=True)
    # An unbound source (a hook's own, as "system") transcodes as a channel
    # of its type with default capabilities.
    source_binding = ready.source_binding or ChannelBinding(
        channel_id=committed.source.channel_id,
        room_id=committed.room_id,
        channel_type=committed.source.channel_type,
    )
    return DeliveryPlan(
        event=committed,
        source_binding=source_binding,
        context=ready.context,
        targets=targets,
        injected=True,
        mutation_hook=mutation_hook,
        fire_after_broadcast=False,
    )
