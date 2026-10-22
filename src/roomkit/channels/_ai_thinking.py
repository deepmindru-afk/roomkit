"""What the agent thinks while it listens, on the AI channel (RFC §6.4).

A channel that has a speak policy and a thinker keeps, per room, the agent's
thought. On an event the policy leaves silent, the thinker rewrites it from the
event's context, one call at a time per room; back with something to say, the
policy is asked again: in time, on the event's own decision; later, when no
event came since, on a decision of its own that runs no turn, for the
``ON_SPEAK_DECISION`` hooks to act on. When the agent speaks, the turn's notes
carry its thought and what it wanted to say is emptied.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import time
from collections.abc import Callable, Coroutine
from functools import partial
from typing import TYPE_CHECKING

from roomkit.speaking.thought import Thought, ThoughtEvent, thought_note

if TYPE_CHECKING:
    from roomkit.channels._ai_callbacks import ThoughtHook
    from roomkit.channels._ai_contract import _AIChannelContract
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.providers.ai.base import AIContext
    from roomkit.speaking.base import SpeakDecision, SpeakPolicy
    from roomkit.speaking.thinker import Thinker
    from roomkit.tools.context import _ToolLoopContext
else:
    _AIChannelContract = object

logger = logging.getLogger("roomkit.channels.ai")


class _RoomMind:
    """One room's thought, and the thinker call that rewrites it: one at a time;
    an event listened to meanwhile is thought about in the next call, from the
    latest context, so a fast exchange never queues stale work."""

    def __init__(self, room_id: str, thinker: Thinker, channel: AIThinkingMixin) -> None:
        self.room_id = room_id
        self.thought = Thought()
        self._thinker = thinker
        self._channel = channel
        self._task: asyncio.Task[None] | None = None
        self._changed = asyncio.Condition()
        self._pending: tuple[AIContext, int, bool, contextvars.Context] | None = None
        """The latest context to think about, how often the agent had spoken
        then and whether it was answering in the room, and the context of the
        turn that asked: its thought is that turn's work, reported under it
        (RFC §10.6.5)."""
        self._asked = 0
        self._done = 0
        self._spoke = 0
        self._late: tuple[int, Callable[[Thought], Coroutine[None, None, None]]] | None = None
        """The ticket whose event was decided before its thought came back, and
        what decides again on it once it does."""

    def think(self, context: AIContext, *, answering: bool = False) -> int:
        """Have the thinker think about *context*, taken while the agent was
        *answering* in the room or not; the ticket :meth:`settled` waits on."""
        self._asked += 1
        self._pending = (context, self._spoke, answering, contextvars.copy_context())
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name=f"thinker:{self.room_id}")
        return self._asked

    async def settled(self, ticket: int, timeout: float) -> bool:
        """Whether the thought covering *ticket* came back within *timeout* seconds."""
        async with self._changed:
            try:
                await asyncio.wait_for(
                    self._changed.wait_for(lambda: self._done >= ticket), timeout
                )
            except TimeoutError:
                # Back as the wait ran out: read in time, or no late decision
                # would ever be owed to it.
                return self._done >= ticket
        return True

    def when_late(
        self, ticket: int, decide: Callable[[Thought], Coroutine[None, None, None]]
    ) -> None:
        """*ticket*'s event was decided without its thought: once the thought
        comes back with something to say, and no later event was given to think
        about, *decide* is called with it."""
        self._late = (ticket, decide)

    def _late_decision(self, ticket: int, thought: Thought) -> Coroutine[None, None, None] | None:
        """The decision owed to *ticket*'s thought, now back: when its event was
        decided without it, it has something to say and no later event came."""
        late = self._late
        if late is None or late[0] != ticket:
            return None  # another event's, or none owed
        self._late = None
        if self._asked != ticket or not thought.want_to_say:
            return None  # a later event reads the thought, or it brings nothing
        return late[1](thought)

    def spoke(self) -> Thought:
        """The agent speaks with its thought: what it wanted to say is on the table.
        Returns the thought it spoke with."""
        thought, self.thought = self.thought, self.thought.said()
        self._spoke += 1
        return thought

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        while self._pending is not None:
            (context, spoke, answering, asked_in), ticket = self._pending, self._asked
            self._pending = None
            started = time.monotonic()
            thought = await self._next(self.thought, context)
            duration_ms = round((time.monotonic() - started) * 1000)
            if self._spoke != spoke or answering:
                # It spoke since the context, or was answering as it was taken:
                # what it wants to say is that answer, not in the context yet.
                thought = thought.said()
            async with self._changed:
                # What the thought replaces is read now: the agent may have
                # spoken during the call, emptying the one the call started from.
                replaced, self.thought = self.thought, thought
                self._done = ticket
                self._changed.notify_all()
            if thought != replaced:
                report = self._channel._report_thought(
                    self.room_id, thought, replaced, duration_ms
                )
                await asyncio.get_running_loop().create_task(report, context=asked_in)
            if (late := self._late_decision(ticket, thought)) is not None:
                await asyncio.get_running_loop().create_task(late, context=asked_in)

    async def _next(self, previous: Thought, context: AIContext) -> Thought:
        try:
            return await self._thinker.think(previous, context)
        except Exception:
            logger.warning("Thinker failed in room %s; thought kept", self.room_id, exc_info=True)
            return previous


class AIThinkingMixin(_AIChannelContract):
    """Keeps the agent's thought per room and runs its thinker while it listens."""

    channel_id: str
    _thinker: Thinker | None
    _think_wait: float
    _thought_hook: ThoughtHook | None
    _minds: dict[str, _RoomMind]
    _active_loops: dict[str, _ToolLoopContext]

    def _store_thinker(
        self, thinker: Thinker | None, wait: float, policy: SpeakPolicy | None
    ) -> None:
        if thinker is not None and policy is None:
            raise ValueError(
                "a thinker needs a speak_policy: it thinks on the events the agent listens to"
            )
        if wait < 0:
            raise ValueError("think_wait must not be negative")
        self._thinker = thinker
        self._think_wait = wait
        self._thought_hook = None
        self._minds = {}

    def _answering_in(self, room_id: str) -> bool:
        """Whether one of the agent's turns is under way in *room_id*."""
        return any(loop.room_id == room_id for loop in self._active_loops.values())

    def _thought_of(self, room_id: str) -> Thought | None:
        """What the agent has in mind in *room_id*; ``None`` without a thinker."""
        if self._thinker is None:
            return None
        mind = self._minds.get(room_id)
        return mind.thought if mind is not None else Thought()

    async def _think_while_listening(
        self,
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
        decision: SpeakDecision,
    ) -> SpeakDecision:
        """The agent stays silent on *event*: it thinks about it, and when the thought
        comes back within the wait with something to say, the policy decides again;
        a final decision waits for no thought."""
        thinker = self._thinker
        if thinker is None:
            return decision
        room_id = context.room.id if context.room else event.room_id
        mind = self._minds.get(room_id) or self._minds.setdefault(
            room_id, _RoomMind(room_id, thinker, self)
        )
        ai_context = await self._thinking_context(event, binding, context)
        if ai_context is None:
            return decision  # BEFORE_AI_GENERATION kept it from the thinker
        ticket = mind.think(ai_context, answering=self._answering_in(room_id))
        late = partial(self._decide_late, event, context)
        if decision.final:
            mind.when_late(ticket, late)
            return decision  # no thought changes it now: nothing to wait for
        if not self._think_wait or not await mind.settled(ticket, self._think_wait):
            mind.when_late(ticket, late)
            return decision
        if not mind.thought.want_to_say:
            return decision
        again = await self._speak_decision(event, context, mind.thought, asked_again=True)
        return again if again is not None else decision

    async def _decide_late(self, event: RoomEvent, context: RoomContext, thought: Thought) -> None:
        """A thought that came back after *event* was decided, with something to
        say: the policy decides again on *event* with it, and the
        ``ON_SPEAK_DECISION`` hooks act on the decision; no turn runs on it, the
        event's turn is over (RFC §6.4)."""
        try:
            await self._speak_decision(event, context, thought, asked_again=True, late=True)
        except Exception:
            logger.warning("Late speak decision failed on %s", event.id, exc_info=True)

    async def _thought_notes(self, room_id: str) -> tuple[str, ...]:
        """The agent speaks on a decided event: its thought joins the turn's notes,
        and what it wanted to say is emptied, a new thought ``ON_THOUGHT`` reports
        (RFC §6.4)."""
        mind = self._minds.get(room_id)
        if mind is None:
            return ()
        spoken = mind.spoke()
        if mind.thought != spoken:
            await self._report_thought(room_id, mind.thought, spoken, None)
        note = thought_note(spoken)
        return (note,) if note else ()

    async def _report_thought(
        self, room_id: str, thought: Thought, previous: Thought, duration_ms: int | None
    ) -> None:
        """Fire ``ON_THOUGHT`` with *thought*, and how long the thinker call that
        brought it took (``None`` without a call)."""
        hook = self._thought_hook
        if hook is None:
            return
        try:
            await hook(ThoughtEvent(room_id, self.channel_id, thought, previous, duration_ms))
        except Exception:
            logger.warning("ON_THOUGHT failed in room %s", room_id, exc_info=True)

    async def _forget_thought(self, room_id: str) -> None:
        """Drop *room_id*'s thought and stop its thinker call."""
        mind = self._minds.pop(room_id, None)
        if mind is not None:
            await mind.close()

    async def _close_minds(self) -> None:
        for mind in self._minds.values():
            await mind.close()
        self._minds.clear()
