"""A supervisor's background work started from its text turn, the same on
its team tool and its per-worker tools (RMK-478, RFC §19.7.3, §23.3).

Both doors are the strategies' background run: the work is bounded by the
supervisor's ``task_timeout``, freed before its outcome is handed back (a
dispatch made in answer starts anew), and its terminal entry says whether the
outcome reached anyone and whether any worker's task completed. A sequential
team runs in the background as in the supervisor's turn: supervised.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIContext, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import until
from tests.test_framework import SimpleChannel
from tests.tool_room import tool_call_in

DOORS = {
    "team-tool": ("delegate_workers", {"strategy": "parallel", "async_delivery": True}),
    "per-worker": ("delegate_to_w1", {"wait_for_result": False}),
}
EVERY_DOOR = pytest.mark.parametrize("door", list(DOORS))


class _Redispatches(MockAIProvider):
    """A supervisor that dispatches again on each outcome handed back, up
    to *limit* dispatches; it records what its tool answered and was told."""

    def __init__(self, tool: str, limit: int) -> None:
        super().__init__()
        self._tool, self._limit = tool, limit
        self.answers: list[str] = []
        self.told: list[str] = []

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        last = context.messages[-1]
        if isinstance(last.content, list):
            results = [p for p in last.content if getattr(p, "type", None) == "tool_result"]
            if results:
                self.answers.append(json.loads(str(results[0].result))["status"])
                return AIResponse(content="On it.", finish_reason="stop")
        if self.answers:
            self.told.append(str(last.content))
        if len(self.answers) >= self._limit:
            return AIResponse(content="Done.", finish_reason="stop")
        n = len(self.answers) + 1
        call = AIToolCall(id=f"c{n}", name=self._tool, arguments={"task": f"t{n}"})
        return AIResponse(content="", finish_reason="tool_calls", tool_calls=[call])


class _Slow(MockAIProvider):
    async def generate(self, context: AIContext) -> AIResponse:
        await asyncio.sleep(5)
        return AIResponse(content="Too late.")


async def _room(
    door: str, boss: _Redispatches, worker: MockAIProvider, **settings: Any
) -> tuple[RoomKit, list[tuple[str, Any, str]]]:
    """Room ``r1`` with the door's supervisor; the run's terminal entries."""
    _, door_settings = DOORS[door]
    supervisor = Supervisor(
        Agent("boss", provider=boss, tool_search=False),
        [Agent("w1", provider=worker)],
        **door_settings,
        **settings,
    )
    kit = RoomKit(max_chain_depth=10)
    kit.register_channel(SimpleChannel("sms1"))
    await kit.create_room(room_id="r1", orchestration=supervisor)
    await kit.attach_channel("r1", "sms1")
    posted: list[tuple[str, Any, str]] = []
    real_post = kit.status_bus.post

    def post(agent_id: str, action: str, status: Any, **kwargs: Any) -> Any:
        if agent_id == "orchestration":
            posted.append((action, status, kwargs.get("detail", "")))
        return real_post(agent_id, action, status, **kwargs)

    kit.status_bus.post = post  # type: ignore[method-assign]
    return kit, posted


async def _ask(kit: RoomKit) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="Analyse X."))
    )


@EVERY_DOOR
async def test_a_dispatch_answering_an_outcome_starts_anew(door: str) -> None:
    boss = _Redispatches(DOORS[door][0], limit=3)
    kit, _ = await _room(door, boss, MockAIProvider(responses=["Findings."]))

    await _ask(kit)
    await until(lambda: len(boss.answers) == 3)
    await kit.close()

    assert boss.answers in (["dispatched"] * 3, ["delegated"] * 3)


@EVERY_DOOR
async def test_an_outcome_no_one_hears_posts_its_run_failed(door: str) -> None:
    boss = _Redispatches(DOORS[door][0], limit=1)
    kit, posted = await _room(door, boss, MockAIProvider(responses=["Findings."]))

    @kit.hook(HookTrigger.BEFORE_DELIVER)
    async def _quiet(event: Any, ctx: Any) -> HookResult:
        return HookResult.block("quiet hours")

    await _ask(kit)
    await until(lambda: bool(posted))
    await kit.close()

    assert [(status, detail) for _, status, detail in posted] == [
        (StatusLevel.FAILED, "not handed back: blocked (quiet hours)")
    ]


@EVERY_DOOR
async def test_a_worker_past_its_bound_is_told_as_not_completed(door: str) -> None:
    boss = _Redispatches(DOORS[door][0], limit=1)
    kit, posted = await _room(door, boss, _Slow(), task_timeout=0.2)

    await _ask(kit)
    await until(lambda: bool(boss.told))
    await kit.close()

    assert "The task timed out after 0.2s." in boss.told[0]
    assert "completed. Share" not in boss.told[0]
    assert [status for _, status, _ in posted] == [StatusLevel.FAILED]


async def _team_delegations(*, async_delivery: bool, voice: bool) -> list[str]:
    """The agents a sequential team's run delegates to, in order."""
    kit = RoomKit()
    boss = Agent("boss", provider=MockAIProvider(responses=["ok"]))
    workers = [Agent(w, provider=MockAIProvider(responses=[w])) for w in ("w1", "w2")]
    delegated: list[str] = []

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def _delegated(event: Any, ctx: Any) -> None:
        delegated.append(event.metadata["agent_id"])

    settings = {"auto_delegate": True} if voice else {}
    strategy = Supervisor(
        boss, workers, strategy="sequential", async_delivery=async_delivery, **settings
    )
    if voice:
        provider = MockRealtimeProvider()
        channel = RealtimeVoiceChannel(
            "voice", provider=provider, transport=MockRealtimeTransport()
        )
        kit.register_channel(channel)
        await kit.create_room(room_id="r1", orchestration=strategy)
        await kit.attach_channel("r1", "voice")
        session = await channel.start_session("r1", "u", "ws")
        await provider.simulate_tool_call(session, "c1", "delegate_workers", {"task": "Do it."})
    else:
        kit.register_channel(boss)
        await kit.create_room(room_id="r1", orchestration=strategy)
        with tool_call_in("r1"):
            await boss._channel_tool_handler("delegate_workers", {"task": "Do it."})
    await until(lambda: len(delegated) >= 3)
    await asyncio.sleep(0.05)
    await kit.close()
    return delegated


@pytest.mark.parametrize(
    ("async_delivery", "voice"),
    [(True, False), (True, True)],
    ids=["team-tool-background", "voice-background"],
)
async def test_a_sequential_team_is_supervised_in_the_background_as_in_its_turn(
    async_delivery: bool, voice: bool
) -> None:
    """The supervisor frames and validates each step whether the team runs
    in its turn or in the background (RFC §19.7.3)."""
    in_turn = await _team_delegations(async_delivery=False, voice=False)

    assert in_turn[0] == "boss"
    assert await _team_delegations(async_delivery=async_delivery, voice=voice) == in_turn
