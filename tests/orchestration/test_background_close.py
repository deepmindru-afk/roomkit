"""``kit.close()`` ends a strategy's background run, on every door that
starts one (RMK-478, RFC §19.7.3, §19.7.4).

The run is cancelled with the kit: its worker's delegation ends cancelled,
its terminal entry is posted failed, nothing is handed back, and no turn of
its starts against the closed kit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.orchestration.strategies.loop import Loop
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.tool_room import tool_call_in


class _Held(MockAIProvider):
    """Answers once *release* is set; *started* says it was asked."""

    def __init__(self, started: asyncio.Event, release: asyncio.Event, **kw: Any) -> None:
        super().__init__(**kw)
        self._started = started
        self._release = release

    async def generate(self, context: AIContext) -> AIResponse:
        self._started.set()
        await self._release.wait()
        return await super().generate(context)


async def _voice_door(kit: RoomKit, strategy: Any, tool: str) -> Callable[[], Awaitable[Any]]:
    """A voice channel's session calling *tool* in room ``r``."""
    provider = MockRealtimeProvider()
    voice = RealtimeVoiceChannel("voice", provider=provider, transport=MockRealtimeTransport())
    kit.register_channel(voice)
    await kit.create_room(room_id="r", orchestration=strategy)
    await kit.attach_channel("r", "voice")
    session = await voice.start_session("r", "u", "ws")
    return lambda: provider.simulate_tool_call(session, "c1", tool, {"task": "Write it."})


async def _supervisor_voice(kit: RoomKit, worker: Agent) -> Callable[[], Awaitable[Any]]:
    boss = Agent("boss", provider=MockAIProvider(responses=["ok"]))
    strategy = Supervisor(
        boss, [worker], strategy="parallel", auto_delegate=True, async_delivery=True
    )
    return await _voice_door(kit, strategy, "delegate_workers")


async def _loop_voice(kit: RoomKit, worker: Agent) -> Callable[[], Awaitable[Any]]:
    editor = Agent("editor", provider=MockAIProvider(responses=["APPROVED"]))
    strategy = Loop(agent=worker, reviewer=editor, async_delivery=True)
    return await _voice_door(kit, strategy, "delegate_loop")


async def _supervisor_tool(kit: RoomKit, worker: Agent) -> Callable[[], Awaitable[Any]]:
    boss = Agent("boss", provider=MockAIProvider(responses=["ok"]))
    kit.register_channel(boss)
    strategy = Supervisor(boss, [worker], strategy="parallel", async_delivery=True)
    await kit.create_room(room_id="r", orchestration=strategy)

    async def call() -> Any:
        with tool_call_in("r"):
            return await boss._channel_tool_handler("delegate_workers", {"task": "Write it."})

    return call


_RUNS = ("_async_run_and_deliver", "_async_loop_and_deliver", "run_in_background")


def _runs_alive() -> list[str]:
    """The strategies' background runs still running."""
    return [
        name
        for task in asyncio.all_tasks()
        if not task.done() and (name := task.get_coro().__qualname__) in _RUNS  # type: ignore[union-attr]
    ]


async def _per_worker(kit: RoomKit, worker: Agent) -> Callable[[], Awaitable[Any]]:
    boss = Agent("boss", provider=MockAIProvider(responses=["ok"]))
    kit.register_channel(boss)
    strategy = Supervisor(boss, [worker], wait_for_result=False)
    await kit.create_room(room_id="r", orchestration=strategy)

    async def call() -> Any:
        with tool_call_in("r"):
            return await boss._channel_tool_handler("delegate_to_worker", {"task": "Write it."})

    return call


DOORS = {
    "supervisor-voice": _supervisor_voice,
    "loop-voice": _loop_voice,
    "supervisor-tool": _supervisor_tool,
    "per-worker": _per_worker,
}


@pytest.mark.parametrize("door", list(DOORS))
async def test_close_ends_a_background_run(door: str) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    worker = Agent("worker", provider=_Held(started, release, responses=["The draft."]))
    kit = RoomKit()
    call = await DOORS[door](kit, worker)
    ends: list[str] = []
    posted: list[tuple[str, Any, str]] = []
    real_post = kit.status_bus.post

    def post(agent_id: str, action: str, status: Any, **kwargs: Any) -> Any:
        posted.append((agent_id, status, kwargs.get("detail", "")))
        return real_post(agent_id, action, status, **kwargs)

    kit.status_bus.post = post  # type: ignore[method-assign]

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def _ended(event: Any, ctx: Any) -> None:
        ends.append(str(event.metadata["task_status"]))

    await call()
    await asyncio.wait_for(started.wait(), 5)
    await asyncio.wait_for(kit.close(), 10)
    runs_left = _runs_alive()
    release.set()
    await asyncio.sleep(0.05)

    assert runs_left == []
    assert ends == ["cancelled"]
    terminal = [p for p in posted if p[0] in ("worker", "orchestration")][-2:]
    assert terminal == [
        ("worker", StatusLevel.FAILED, "cancelled"),
        ("orchestration", StatusLevel.FAILED, "cancelled"),
    ]
    assert worker._provider.calls == []  # type: ignore[attr-defined]
