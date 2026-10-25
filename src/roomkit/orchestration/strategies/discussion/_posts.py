"""The task results a room's discussion posts as its agents' messages (RFC §19.7.5 rule 19).

A result handed back with ``hand_back="post"`` takes a turn of its own, or
waits for the floor, and its turn posts it without a generation. From when
the discussion takes it until it is committed, the result is listed with the
room's private state in the store, out of the room's metadata: its text is
read nowhere the posted message would not be. Whichever process
uninstalls the discussion posts every listed result before it detaches the
agents, the ones another process holds or is committing included. A post is
keyed by the result's id, the id it was listed under, so whichever post
comes second is a replay (RFC §13.4).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from roomkit.core.exceptions import ChannelNotFoundError
from roomkit.core.task_utils import hold_task
from roomkit.models.enums import ChannelType, EventStatus, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.tasks.posting import send_posted

if TYPE_CHECKING:
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.orchestration.discussion")

RESULTS_KEY = "_speak_queue_results"
"""The key a room's listed results are kept under, in its private state."""

_FLOOR_POLL = 0.05
"""How often a result to post looks whether the floor is free."""


class PostedResult(BaseModel):
    """What a task's result posts as its agent's message."""

    body: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    channel_type: ChannelType
    """The agent's channel type, the posted message's source."""


class ListedResult(BaseModel):
    """A task's result the discussion took and has not committed yet."""

    id: str
    """The id it was taken under, which keys its post."""
    agent: str
    depth: int
    result: PostedResult

    @classmethod
    def of(cls, event: RoomEvent) -> ListedResult | None:
        """*event*, a result to post, as it is listed; None for one that is
        not text, which only the process holding it posts."""
        if not isinstance(event.content, TextContent):
            return None
        result = PostedResult(
            body=event.content.body,
            metadata=event.metadata or {},
            channel_type=event.source.channel_type,
        )
        return cls(
            id=event.id, agent=event.source.channel_id, depth=event.chain_depth, result=result
        )

    def event(self, room_id: str) -> RoomEvent:
        """The event its holder would post, under the same id."""
        return RoomEvent(
            id=self.id,
            room_id=room_id,
            type=EventType.MESSAGE,
            source=EventSource(channel_id=self.agent, channel_type=self.result.channel_type),
            content=TextContent(body=self.result.body),
            chain_depth=self.depth,
            status=EventStatus.DELIVERED,
            metadata=self.result.metadata,
            addressed_to=[],
        )


class ResultPosts:
    """One room's task results to post, in this process and as listed."""

    def __init__(self, kit: Any, room_id: str, floor_free: Callable[[], bool]) -> None:
        self._kit = kit
        self._room_id = room_id
        self._floor_free = floor_free
        self._unposted: dict[str, RoomEvent] = {}
        """Results waiting for the floor here, by id, until a post takes them."""

    async def keep(self, posted: RoomEvent) -> None:
        """List *posted*, a result the discussion takes, for any process that
        uninstalls it to post. Under the room lock, the caller's."""
        listed = ListedResult.of(posted)
        if listed is None:
            return
        results = await _load(self._kit.store, self._room_id)
        if all(r.id != listed.id for r in results):
            await _save(self._kit.store, self._room_id, [*results, listed])

    async def take_listed(self) -> list[ListedResult]:
        """The results listed, no longer listed: the discussion is being
        uninstalled. Under the room lock, the caller's."""
        results = await _load(self._kit.store, self._room_id)
        if results:
            await _save(self._kit.store, self._room_id, [])
        return results

    def post_later(self, posted: RoomEvent) -> None:
        """Post a task's result its own turn will not post, once no turn runs,
        in a task the kit holds: a closing kit posts nothing, and an uninstall
        posts it itself, before the agents leave (rule 19)."""
        if self._kit._closed:
            return
        self._unposted[posted.id] = posted
        # A context of its own: the room lock may be held here, and is
        # reentrant per context.
        hold_task(self._kit, self._post_when_free(posted.id), context=contextvars.Context())

    def post_held(self, held: dict[str, RoomEvent], posting: RoomEvent | None) -> None:
        """Post every task result *held* holds for a queued turn, the queue
        being gone: another process uninstalled the discussion (rule 19).
        *posting*, the one a post turn is committing, goes on by itself."""
        for event_id, event in list(held.items()):
            if event.type != EventType.MESSAGE:
                continue  # an instruction: its turn is the agent's, now gone
            held.pop(event_id, None)
            if posting is None or event_id != posting.id:
                self.post_later(event)

    async def _post_when_free(self, event_id: str) -> None:
        while not self._floor_free():
            await asyncio.sleep(_FLOOR_POLL)
        posted = self._unposted.pop(event_id, None)
        if posted is not None:  # else an uninstall posted it
            await self.post(posted)

    async def post(self, posted: RoomEvent) -> None:
        """Commit *posted*, a task's result, as its agent's message, then take
        it off the list; one cut or failed on the way stays listed, for an
        uninstall to post, as one a closing kit does not post does. Run in a
        task the kit holds."""
        if self._kit._closed:
            return
        await self._commit(posted)
        async with self._kit._lock_manager.locked(self._room_id):
            results = await _load(self._kit.store, self._room_id)
            kept = [r for r in results if r.id != posted.id]
            if len(kept) != len(results):
                await _save(self._kit.store, self._room_id, kept)

    async def post_all(self, listed: list[ListedResult]) -> None:
        """Post *listed*, the results the discussion being uninstalled listed,
        and those waiting for the floor here, in the caller's task: it holds
        the room lock or takes it, so their agents are still bound (rule 19).
        A post another process is committing meanwhile, or this one's post
        turn, comes second and is a replay. A failed post is logged and never
        blocks the uninstall."""
        held = {r.id: r.event(self._room_id) for r in listed}
        for event_id, event in self._unposted.items():
            held.setdefault(event_id, event)
        self._unposted = {}
        try:
            while held:
                posted = held.pop(next(iter(held)))
                try:
                    await self._commit(posted)
                except Exception:
                    logger.exception(
                        "Task result %s in room %s was not posted", posted.id, self._room_id
                    )
        finally:
            # Cut on the way: the kit posts the rest, as it would have.
            for posted in held.values():
                self.post_later(posted)

    async def _commit(self, posted: RoomEvent) -> None:
        """Commit *posted*; one whose agent was detached before any post of it
        is logged, lost."""
        try:
            await send_posted(self._kit, posted)
        except ChannelNotFoundError:
            logger.warning(
                "task result %s for %s in %s not posted: agent detached",
                posted.id,
                posted.source.channel_id,
                self._room_id,
            )


async def _load(store: ConversationStore, room_id: str) -> list[ListedResult]:
    """The results listed for *room_id*, in its private state."""
    value = await store.get_private_state(room_id, RESULTS_KEY)
    return [ListedResult.model_validate(r) for r in (value or {}).get("results", [])]


async def _save(store: ConversationStore, room_id: str, results: list[ListedResult]) -> None:
    """List *results* for *room_id*, none removing the list. Under the room
    lock, the caller's."""
    value = {"results": [r.model_dump(mode="json") for r in results]} if results else None
    await store.set_private_state(room_id, RESULTS_KEY, value)
