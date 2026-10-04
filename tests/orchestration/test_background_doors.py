"""A strategy's background door, the same on the supervisor's and the Loop's
voice tool (RMK-462, RFC §19.7.3, §19.7.4, §23.3 step 8).

One run per room, whichever voice channel's session asked for it; the
outcome is told in the session that made the call, as an instruction; and
the run's terminal entry says whether the outcome reached anyone.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.orchestration.strategies import loop as loop_module
from roomkit.orchestration.strategies.loop import Loop
from roomkit.orchestration.strategies.supervisor import Supervisor, _install_auto
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import until

DOORS = {
    "loop": (
        "delegate_loop",
        lambda: Loop(
            agent=Agent("writer", provider=MockAIProvider(responses=["The draft."])),
            reviewer=Agent("editor", provider=MockAIProvider(responses=["APPROVED"])),
            async_delivery=True,
        ),
    ),
    "supervisor": (
        "delegate_workers",
        lambda: Supervisor(
            supervisor=Agent("boss", provider=MockAIProvider(responses=["ok"])),
            workers=[Agent("analyst", provider=MockAIProvider(responses=["The draft."]))],
            strategy="parallel",
            auto_delegate=True,
            async_delivery=True,
        ),
    ),
}
BOTH = pytest.mark.parametrize("door", list(DOORS))


async def _room(
    door: str, voices: tuple[str, ...], sessions_per_voice: int = 1
) -> tuple[RoomKit, MockRealtimeProvider, list[Any]]:
    """A room with the door's strategy, *voices* attached (each with its own
    provider; the first one's is returned), and sessions on each."""
    providers = [MockRealtimeProvider() for _ in voices]
    provider = providers[0]
    kit = RoomKit()
    channels = [
        RealtimeVoiceChannel(v, provider=p, transport=MockRealtimeTransport())
        for v, p in zip(voices, providers, strict=True)
    ]
    for channel in channels:
        kit.register_channel(channel)
    await kit.create_room(room_id="r", orchestration=DOORS[door][1]())
    sessions = []
    for channel in channels:
        await kit.attach_channel("r", channel.channel_id)
        for n in range(sessions_per_voice):
            sessions.append(await channel.start_session("r", f"u{n}", "ws"))
    return kit, provider, sessions


def _told(provider: MockRealtimeProvider) -> list[tuple[str, str]]:
    """Each background outcome injected: the session told, and the intent."""
    return [
        (c.args["session_id"], c.args["role"])
        for c in provider.calls
        if c.method == "inject_text" and "[Your background" in str(c.args.get("text"))
    ]


@BOTH
async def test_only_the_session_that_called_is_told(door: str) -> None:
    kit, provider, (caller, other) = await _room(door, ("voice",), sessions_per_voice=2)

    await provider.simulate_tool_call(caller, "c1", DOORS[door][0], {"task": "Write it."})
    await until(lambda: bool(_told(provider)))
    await asyncio.sleep(0.05)
    await kit.close()

    assert _told(provider) == [(caller.id, "system")]


def _hold_runs(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], asyncio.Event]:
    """Runs that wait for *release* before freeing their room."""
    started: list[str] = []
    release = asyncio.Event()

    async def held(**kwargs: Any) -> None:
        started.append(kwargs["room_id"])
        await release.wait()
        on_done: Callable[..., None] = kwargs["on_done"]
        on_done(success=True) if "supervisor_id" in kwargs else on_done()

    monkeypatch.setattr(loop_module, "_async_loop_and_deliver", held)
    monkeypatch.setattr(_install_auto, "_async_run_and_deliver", held)
    return started, release


@BOTH
async def test_one_run_per_room_whichever_voice_channel_asks(
    door: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release = _hold_runs(monkeypatch)
    kit, provider_a, (on_a, on_b) = await _room(door, ("voice-a", "voice-b"))
    provider_b = kit.get_channel("voice-b")._provider  # type: ignore[union-attr]
    tool = DOORS[door][0]

    await provider_a.simulate_tool_call(on_a, "c1", tool, {"task": "Write it."})
    await until(lambda: len(provider_a.tool_results) == 1)
    await provider_b.simulate_tool_call(on_b, "c2", tool, {"task": "Write it again."})
    await until(lambda: len(provider_b.tool_results) == 1)
    release.set()
    await asyncio.sleep(0.05)
    await kit.close()

    assert json.loads(provider_a.tool_results[0][2])["status"] in ("started", "dispatched")
    assert json.loads(provider_b.tool_results[0][2])["status"] == "already_running"
    assert started == ["r"]


async def test_a_loop_its_producer_stopped_posts_a_failed_terminal_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stopped(**kwargs: Any) -> Any:
        return loop_module._LoopOutcome(stopped="producer_failed")

    monkeypatch.setattr(loop_module, "_execute_loop", stopped)
    kit, provider, (session,) = await _room("loop", ("voice",))
    posted: list[tuple[str, Any, str]] = []
    real_post = kit.status_bus.post

    def post(agent_id: str, action: str, status: Any, **kwargs: Any) -> Any:
        if agent_id == "orchestration":
            posted.append((action, status, kwargs.get("detail", "")))
        return real_post(agent_id, action, status, **kwargs)

    kit.status_bus.post = post  # type: ignore[method-assign]
    await provider.simulate_tool_call(session, "c1", "delegate_loop", {"task": "Write it."})
    await until(lambda: bool(posted))
    await kit.close()

    assert posted == [("loop", StatusLevel.FAILED, "producer_failed")]
    assert _told(provider) == [(session.id, "system")]
