"""Supervisor reviews through the real tool loop, one room isolated from another.

A supervisor is one Agent serving every room. Its review hands the verdict back
through the ``submit_verdict`` tool the delegation installs on that shared
channel, so two reviews at once must neither read each other's verdict nor leave
the channel changed once both end (RMK-246 review).
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import Agent, RoomKit
from roomkit.orchestration.strategies.supervisor.supervised import _supervisor_review
from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    AIToolCall,
    AIToolResultPart,
)


class _JudgeByOutput(AIProvider):
    """Approves the output marked GOOD, rejects the one marked BAD, and answers
    each through a ``submit_verdict`` call; after the call, a closing line.

    The first turn of each review pauses, so two reviews started together are
    both mid-turn at once.
    """

    def __init__(self, *, streaming: bool = False) -> None:
        self.tools_seen: list[list[str]] = []
        self._streaming = streaming

    @property
    def model_name(self) -> str:
        return "judge"

    @property
    def supports_structured_streaming(self) -> bool:
        return self._streaming

    async def generate(self, context: AIContext) -> AIResponse:
        if any(
            isinstance(part, AIToolResultPart)
            for message in context.messages
            if isinstance(message.content, list)
            for part in message.content
        ):
            return AIResponse(content="Verdict submitted.")
        self.tools_seen.append([tool.name for tool in context.tools])
        await asyncio.sleep(0.05)
        prompt = " ".join(str(message.content) for message in context.messages)
        approved = "GOOD" in prompt
        verdict = {
            "approved": approved,
            "feedback": "" if approved else "BAD: add a source",
            "next_task": "",
        }
        return AIResponse(
            content="",
            tool_calls=[AIToolCall(id="call-1", name="submit_verdict", arguments=verdict)],
        )


async def _review(kit: RoomKit, boss: Agent, worker: Agent, room: str, output: str) -> Any:
    return await _supervisor_review(
        kit,
        boss,
        room,
        goal="the largest city of Québec",
        worker=worker,
        output=output,
        next_worker=None,
        share_channels=None,
        task_timeout=10.0,
    )


async def _kit_with_supervisor(*, streaming: bool) -> tuple[RoomKit, Agent, Agent, _JudgeByOutput]:
    kit = RoomKit()
    judge = _JudgeByOutput(streaming=streaming)
    boss = Agent("boss", provider=judge, role="Supervisor")
    worker = Agent("w1", provider=judge, role="Researcher")
    kit.register_channel(boss)
    kit.register_channel(worker)
    for room in ("room-a", "room-b"):
        await kit.create_room(room_id=room)
    return kit, boss, worker, judge


class TestReviewThroughTheToolLoop:
    async def test_a_review_reads_the_verdict_its_room_submitted(self, streaming: bool) -> None:
        kit, boss, worker, _judge = await _kit_with_supervisor(streaming=streaming)

        verdict = await _review(kit, boss, worker, "room-a", "GOOD: Montréal, StatCan")

        assert verdict == {"approved": True, "feedback": "", "next_task": None}

    async def test_two_reviews_at_once_each_read_their_own_verdict(self, streaming: bool) -> None:
        kit, boss, worker, judge = await _kit_with_supervisor(streaming=streaming)
        handler_before = boss.tool_handler
        tools_before = list(boss._injected_tools)

        good, bad = await asyncio.gather(
            _review(kit, boss, worker, "room-a", "GOOD: Montréal, StatCan"),
            _review(kit, boss, worker, "room-b", "BAD: no idea"),
        )

        assert good["approved"] is True
        assert bad["approved"] is False
        assert bad["feedback"] == "BAD: add a source"
        # The verdict tool was offered once per turn, never twice.
        assert all(names.count("submit_verdict") == 1 for names in judge.tools_seen)
        # And the shared channel is left as it was found.
        assert boss.tool_handler is handler_before
        assert boss._injected_tools == tools_before
        assert boss._room_tools == {}
