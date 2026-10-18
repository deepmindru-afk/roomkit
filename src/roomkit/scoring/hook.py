"""Hook that runs conversation scorers on AI responses.

Usage::

    from roomkit.scoring import ScoringHook, MockScorer

    hook = ScoringHook(scorers=[MockScorer()])
    hook.attach(kit)  # registers ON_AI_RESPONSE on the kit's turn executor
    hook.attach(executor)  # or on an executor a host built

    # Scores are stored as Observations by the room's owner, through the
    # turn's host, and accessible via hook.recent_scores
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit._media import content_text
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.models.task import Observation
from roomkit.models.tool_call import AIResponseEvent
from roomkit.scoring.base import ConversationScorer, Score
from roomkit.turns.executor import TurnExecutor
from roomkit.turns.host import current_turn_host

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.scoring")


class ScoringHook:
    """Runs conversation scorers on every AI response.

    Attach to a :class:`~roomkit.RoomKit` instance to automatically
    score AI responses via the ``ON_AI_RESPONSE`` hook.  Scores are
    stored as :class:`~roomkit.models.task.Observation` objects through
    the turn's host, by the room's owner, wherever the turn runs (RFC
    §10.6.5), and kept in a bounded in-memory buffer for quick access.

    Parameters:
        scorers: List of :class:`ConversationScorer` implementations.
        store: Ignored: an executor never writes the owner's store itself;
            kept so existing calls still construct.
        max_recent: Maximum number of scores to keep in the in-memory buffer.
    """

    def __init__(
        self,
        scorers: list[ConversationScorer],
        *,
        store: ConversationStore | None = None,
        max_recent: int = 100,
    ) -> None:
        self._scorers = scorers
        self._store: ConversationStore | None = store
        self.recent_scores: deque[dict[str, Any]] = deque(maxlen=max_recent)

    def attach(self, target: RoomKit | TurnExecutor) -> None:
        """Register the scoring hook on a turn executor, where ON_AI_RESPONSE
        fires (RFC §10.6.6): a kit's own, or one a host built. Scores reach
        the room's owner through the turn's host."""
        executor = target if isinstance(target, TurnExecutor) else target.turn_executor
        decorator = executor.hook(
            HookTrigger.ON_AI_RESPONSE,
            execution=HookExecution.ASYNC,
            name="scoring_hook",
        )
        decorator(self._on_response)

    async def _on_response(self, event: Any, context: Any) -> None:
        """Handle ON_AI_RESPONSE hook — run all scorers."""
        if not isinstance(event, AIResponseEvent):
            return

        query = self._extract_last_user_query(event, context)

        raw_results = await asyncio.gather(
            *[
                s.score(
                    response_content=event.response_content,
                    query=query,
                    room_id=event.room_id or "",
                    channel_id=event.channel_id,
                    usage=event.usage,
                    thinking=event.thinking,
                )
                for s in self._scorers
            ],
            return_exceptions=True,
        )

        all_scores: list[tuple[str, Score]] = []
        for i, result in enumerate(raw_results):
            if isinstance(result, BaseException):
                logger.warning("Scorer %s failed: %s", self._scorers[i].name, result)
                continue
            for score in result:
                all_scores.append((self._scorers[i].name, score))

        if not all_scores:
            return

        await self._persist_scores(event, all_scores)
        self._buffer_scores(event, all_scores)

    async def _persist_scores(
        self,
        event: AIResponseEvent,
        scores: list[tuple[str, Score]],
    ) -> None:
        """Store scores as Observations, through the turn's host."""
        if not event.room_id:
            return
        for scorer_name, score in scores:
            obs = Observation(
                id=uuid4().hex,
                room_id=event.room_id,
                channel_id=event.channel_id,
                content=f"[{score.dimension}] {score.value:.2f}: {score.reason}",
                category=f"score:{score.dimension}",
                confidence=score.value,
                metadata={
                    "scorer": scorer_name,
                    "dimension": score.dimension,
                    "score_value": score.value,
                    "reason": score.reason,
                    "latency_ms": event.latency_ms,
                    "streaming": event.streaming,
                    **score.metadata,
                },
            )
            try:
                # A scorer writes through its turn's host, during the turn and
                # after it (RFC §10.6.5): an executor never writes the owner's
                # store itself.
                host = current_turn_host()
                if host is None:
                    logger.warning(
                        "No room owner to store a score of room %s: dropped", event.room_id
                    )
                    return
                await host.add_observation(obs)
            except Exception:
                logger.warning("Failed to store score observation", exc_info=True)

    def _buffer_scores(
        self,
        event: AIResponseEvent,
        scores: list[tuple[str, Score]],
    ) -> None:
        """Add scores to the in-memory buffer for quick access."""
        for scorer_name, score in scores:
            self.recent_scores.append(
                {
                    "room_id": event.room_id,
                    "channel_id": event.channel_id,
                    "scorer": scorer_name,
                    "dimension": score.dimension,
                    "value": score.value,
                    "reason": score.reason,
                    "latency_ms": event.latency_ms,
                }
            )

    @staticmethod
    def _extract_last_user_query(event: AIResponseEvent, context: Any) -> str:
        """Extract the last user message from context as query for scoring."""
        if not hasattr(context, "recent_events") or not context.recent_events:
            return ""
        for e in reversed(context.recent_events):
            if e.source.channel_id == event.channel_id:
                continue
            if text := content_text(e.content):
                return text
        return ""

    async def close(self) -> None:
        """Close all scorers."""
        for s in self._scorers:
            try:
                await s.close()
            except Exception:
                logger.debug("Scorer %s close failed", s.name, exc_info=True)
