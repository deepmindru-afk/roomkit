"""Shared event-visibility resolution.

A single source of truth for how an event's ``visibility`` scope maps to a
target channel binding, used by both the broadcast router and the streaming
target selector so the two can never drift — and, since RFC §7.5 rule 8, by
the reconstruction of a channel's history, which must answer the same question
one turn later that the broadcast answered at delivery.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from roomkit.core._changes import withheld
from roomkit.models._visibility import KEYWORDS
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, Visibility
from roomkit.models.event import RoomEvent
from roomkit.models.store_filter import received_events

logger = logging.getLogger("roomkit.framework")

_CATEGORIES = {
    Visibility.TRANSPORT: ChannelCategory.TRANSPORT,
    Visibility.INTELLIGENCE: ChannelCategory.INTELLIGENCE,
}


def intersect_visibility(
    first: str, second: str, category_of: Callable[[str], ChannelCategory | None]
) -> str:
    """The visibility that includes exactly the channels both include (RFC
    §7.3).

    ``all`` is neutral; ``internal`` absorbs and keeps the mark of a
    framework record; ``none`` absorbs; ``transport`` and ``intelligence``
    exclude each other; two lists intersect. A category intersected with a
    list keeps the listed channels of that category, which *category_of*
    tells from the room's bindings; a channel it does not know matches
    nothing. An empty result is ``none``.
    """
    a, b = normal_visibility(first), normal_visibility(second)
    if a == Visibility.ALL:
        return b
    if b == Visibility.ALL:
        return a
    if Visibility.INTERNAL in (a, b):
        return Visibility.INTERNAL
    if Visibility.NONE in (a, b):
        return Visibility.NONE
    if a in _CATEGORIES and b in _CATEGORIES:
        return a if a == b else Visibility.NONE
    if a in _CATEGORIES:
        return _listing(i for i in _ids(b) if category_of(i) == _CATEGORIES[a])
    if b in _CATEGORIES:
        return _listing(i for i in _ids(a) if category_of(i) == _CATEGORIES[b])
    return _listing(_ids(a) & _ids(b))


def narrower_scope(
    first: str | None, second: str | None, bindings: Iterable[ChannelBinding]
) -> str | None:
    """The answer scope within both *first* and *second* (RFC §5.12, §7.3):
    ``None``, no restriction, on either side gives the other; two scopes give
    their intersection, a list matched against the categories of the room's
    *bindings*. Never wider than either."""
    if first is None:
        return second
    if second is None:
        return first
    categories = {binding.channel_id: binding.category for binding in bindings}
    return intersect_visibility(first, second, categories.get)


def committed_visibility(
    visibility: str, source_binding: ChannelBinding | None, bindings: Iterable[ChannelBinding]
) -> str:
    """The visibility an event is committed with: *visibility*, its own,
    intersected with its source binding's (RFC §7.3). A list is matched
    against the categories of the room's *bindings*. An event with no source
    binding, which only the framework or a hook writes, keeps its own."""
    if source_binding is None:
        return visibility
    categories = {binding.channel_id: binding.category for binding in bindings}
    categories[source_binding.channel_id] = source_binding.category
    return intersect_visibility(visibility, source_binding.visibility, categories.get)


def with_committed_visibility(
    event: RoomEvent, source_binding: ChannelBinding | None, bindings: Iterable[ChannelBinding]
) -> RoomEvent:
    """*event* carrying the visibility it is committed with
    (:func:`committed_visibility`), whichever path commits it."""
    stamped = committed_visibility(event.visibility, source_binding, bindings)
    if stamped == event.visibility:
        return event
    return event.model_copy(update={"visibility": stamped})


def narrowed_to_target(
    event: RoomEvent, target: RoomEvent | None, bindings: Iterable[ChannelBinding]
) -> RoomEvent:
    """*event*, an accepted edit or delete of *target*, with its visibility
    narrowed to its target's (RFC §10.3 state update 4): a correction of a
    message one channel alone saw reaches no other. A list is matched
    against the categories of the room's *bindings*; *event* as it is with
    no target."""
    if target is None:
        return event
    categories = {binding.channel_id: binding.category for binding in bindings}
    visibility = intersect_visibility(event.visibility, target.visibility, categories.get)
    if visibility == event.visibility:
        return event
    return event.model_copy(update={"visibility": visibility})


def normal_visibility(visibility: str) -> str:
    """*visibility* in the normal form (RFC §7.3), leniently: a value
    stored before the grammar was enforced (an unsorted list, an empty item)
    is read as the channels it names, ``none`` when it names none."""
    stripped = visibility.strip()
    if stripped in KEYWORDS:
        return stripped
    return _listing(_ids(stripped))


def _ids(visibility: str) -> set[str]:
    return {item.strip() for item in visibility.split(",") if item.strip()}


def _listing(ids: Iterable[str]) -> str:
    listed = sorted(set(ids))
    return ",".join(listed) if listed else Visibility.NONE


def without_framework_mark(event: RoomEvent) -> RoomEvent:
    """*event*, a channel's answer or a row of its stream, as the pipeline
    commits it: one carrying ``internal``, which only the framework writes,
    with ``none`` and a warning logged (RFC §7.3, §8.5). It is stored and
    reaches no channel, as asked, without the mark of the framework's own
    records."""
    if event.visibility != Visibility.INTERNAL:
        return event
    logger.warning(
        "Channel %s answered with the visibility 'internal', which only the framework "
        "writes: committed with 'none'",
        event.source.channel_id,
        extra={"room_id": event.room_id, "channel_id": event.source.channel_id},
    )
    return event.model_copy(update={"visibility": Visibility.NONE})


def visibility_allows(visibility: str, target_binding: ChannelBinding) -> bool:
    """Return whether an event with ``visibility`` reaches ``target_binding``.

    ``visibility`` is either a well-known scope keyword (see
    :class:`~roomkit.models.enums.Visibility`) or a channel-id spec — a single
    channel id, or a comma-separated list of ids.
    """
    if visibility == Visibility.ALL:
        return True
    if visibility == Visibility.NONE:
        return False
    if visibility == Visibility.INTERNAL:
        # Framework-internal events (delegation, system, handoff) are never
        # delivered to channels — they live only in stored room history.
        return False
    if visibility == Visibility.TRANSPORT:
        return target_binding.category == ChannelCategory.TRANSPORT
    if visibility == Visibility.INTELLIGENCE:
        return target_binding.category == ChannelCategory.INTELLIGENCE
    return target_binding.channel_id in _ids(visibility)


def visible_events(context: RoomContext, channel_id: str) -> list[RoomEvent]:
    """``context.recent_events`` as ``channel_id`` is allowed to know it.

    RFC §7.5 rule 8: an event visibility withheld from a channel at delivery
    MUST NOT come back to it through history the framework rebuilds on its
    behalf. Withholding an event at broadcast and returning it a turn later as
    context enforces nothing.

    A channel always keeps its **own** accepted events. Rule 5 skips them at
    delivery because the channel produced them, not because it may not know
    them — dropping them here would erase an assistant's own turns from its
    own prompt whenever it is bound with a narrow visibility (the §7.4
    assistant pattern).

    A channel with no binding in this room is handed none of its events.

    The event's ``visibility`` is the whole answer: the intersection of its
    own and its source binding's, stamped at its commit (RFC §7.3), so a
    binding changed later does not change what its past events reached.

    An event stored BLOCKED was delivered to nobody — a hook refused it (§10.1
    step 10), its source could not write, read-only or muted (step 11, §7.5
    rule 2), the chain-depth cap or the reentry cap took it (§8.3) — and
    reaches nobody here either, the channel that produced it included: the
    room refused that turn, and an agent that reads its own refused answer as
    history continues from a turn nobody received. A muted agent therefore
    keeps tracking the room while muted, and loses its own silenced answers
    from its prompt, on unmute too. The record stays in the timeline for host
    code and audit, behind ``EventFilter(include_blocked=True)`` on a timeline
    read; the same predicate,
    :func:`~roomkit.models.store_filter.received_events`, decides both. The
    own-events exception above is about what a channel may *know*, and a
    refused turn is not that.

    Answers the visibility half of §7.5 rule 1 only. Access is enforced where
    events are delivered (:meth:`EventRouter._filter_targets` drops WRITE_ONLY
    and NONE), which is why a channel that may not read never reaches a context
    rebuild in the first place; a caller reconstructing history by itself owes
    that check. READ_ONLY and muted channels do read, and keep the history they
    were handed; their own refused turns are the one thing they lose, as above.

    An event an accepted DELETE removed stays, its content withheld
    (:func:`~roomkit.core._changes.withheld`, RFC §10.3): the delete that
    follows names an event the reader holds, and nothing of what was said
    reaches it, nor of an edit of it.
    """
    reader = next((b for b in context.bindings if b.channel_id == channel_id), None)
    if reader is None:
        return []
    return [
        e
        for e in withheld(received_events(context.recent_events))
        if e.source.channel_id == channel_id or visibility_allows(e.visibility, reader)
    ]
