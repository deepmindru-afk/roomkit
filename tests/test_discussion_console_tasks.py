"""The discussion console: the answer an agent is writing, the room's tasks, and
who wrote a posted result (RMK-671)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput

from roomkit import Agent, Discussion, RoomKit, WebSocketChannel
from roomkit.console import AgentCard, DiscussionConsole
from roomkit.console._discussion_view import DiscussionView
from roomkit.orchestration.strategies.discussion import (
    SpeakQueue,
    SpeakQueueChange,
    SpeakQueueEvent,
)
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from tests.orchestration.test_strategy_discussion import _Chunked
from tests.test_framework import SimpleChannel

CARDS = (AgentCard("dev", model="m-1"), AgentCard("sre"))


def _room(view: DiscussionView) -> str:
    return "\n".join(line for _style, line in view.lines)


def _shown(view: DiscussionView) -> str:
    return "\n".join(line for _style, line in view.shown_lines())


def _panel(view: DiscussionView) -> str:
    return "".join(text for _style, text in view.agent_fragments())


# -- The view --


def test_the_answer_being_written_shows_until_it_is_said() -> None:
    view = DiscussionView(CARDS, you="ops")
    view.live_start("dev")
    view.live_text("The tax ")
    view.live_text("engine fails.")

    assert "@dev · m-1 writing…" in _shown(view)
    assert "  The tax engine fails.▍" in _shown(view)
    assert "writing" not in _room(view)

    view.agent("dev", None, "The tax engine fails.")
    assert "▍" not in _shown(view)
    view.live_end()
    assert view.live is None


def test_a_running_task_shows_on_the_card_of_the_agent_it_is_for() -> None:
    view = DiscussionView(CARDS, you="ops")
    view.task_started("task-abc123def456", "dev-code", "read the tax engine's source")
    view.task_for("task-abc123def456", "dev")

    assert "task abc123 started: @dev-code — read the tax engine's source" in _room(view)
    panel = _panel(view)
    assert panel.index("⧗ 1 task running") > panel.index("@dev")
    assert panel.index("⧗ 1 task running") < panel.index("@sre")

    view.task_ended("task-abc123def456", "completed", 31.2)
    assert "task abc123 completed after 31 s (@dev)" in _room(view)
    assert "task running" not in _panel(view)


def test_a_posted_result_names_the_worker_that_wrote_it() -> None:
    view = DiscussionView(CARDS, you="ops")
    view.agent("dev", [], "Result of the task “x”:\n\nFindings.", by="dev-code")
    room = _room(view)
    assert "@dev · a task's result, by dev-code → nobody" in room
    assert "m-1" not in room


def _asked(*asked: tuple[str, str]) -> SpeakQueueEvent:
    return SpeakQueueEvent(
        room_id="r1",
        queue=SpeakQueue(
            speaking=None,
            queue=(),
            listening=frozenset(),
            asked=asked,
            waiting=True,
            over=False,
        ),
        change=SpeakQueueChange.WAITING,
        channel_ids=(),
    )


def test_a_person_s_label_never_reads_with_two_at_signs() -> None:
    view = DiscussionView(CARDS, you="ops")
    view.queue_changed(_asked(("sre", "@ops"), ("dev", "@alice")))
    room = _room(view)
    assert ">>> @sre is asking you" in room
    assert ">>> @dev is asking @alice" in room
    assert "@@" not in room


# -- The console on a room --


@pytest.fixture
def pipe() -> Iterator[PipeInput]:
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        yield pipe


async def _until(predicate: Any, timeout: float = 5) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


class _Slow(MockAIProvider):
    def __init__(self, answer: str, seconds: float) -> None:
        super().__init__(ai_responses=[AIResponse(content=answer)])
        self.seconds = seconds

    async def generate(self, context: AIContext) -> AIResponse:
        await asyncio.sleep(self.seconds)
        return await super().generate(context)


async def _console(
    kit: RoomKit, pipe: PipeInput, **options: Any
) -> tuple[DiscussionConsole, asyncio.Task[None]]:
    console = DiscussionConsole(kit, "r1", channel_id="ops", **options)
    running = asyncio.create_task(console.run())
    await _until(lambda: console._view is not None)
    return console, running


async def test_the_console_shows_an_answer_as_it_streams(pipe: PipeInput) -> None:
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("ops"))
    dev = Agent("dev", provider=_Chunked("The engine fails."))
    await kit.create_room(room_id="r1", orchestration=Discussion([dev]))
    await kit.attach_channel("r1", "ops")
    console, running = await _console(kit, pipe)
    deltas: list[str] = []
    live_text = console.view.live_text

    def record(delta: str, stream: str = "") -> None:
        deltas.append(delta)
        live_text(delta, stream)

    console.view.live_text = record  # type: ignore[method-assign]

    pipe.send_text("@dev status?\r")
    await _until(lambda: "The engine fails." in _room(console.view))
    await _until(lambda: console.view.live is None)

    assert "".join(deltas) == "The engine fails."
    console.exit()
    await asyncio.wait_for(running, 5)
    # Closed, it leaves no connection behind on the person's channel.
    assert kit.channels["ops"].connection_count == 0  # type: ignore[attr-defined]
    await kit.close()


async def _tasks_room(kit: RoomKit) -> None:
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(Agent("helper", provider=_Slow("Findings.", 5.0)))
    dev = Agent("dev", provider=MockAIProvider(ai_responses=[AIResponse(content="ok")]))
    sre = Agent("sre", provider=MockAIProvider(ai_responses=[AIResponse(content="ok")]))
    kit.setup_background(dev, worker="helper")
    await kit.create_room(room_id="r1", orchestration=Discussion([dev, sre]))
    await kit.attach_channel("r1", "ops")


async def test_the_person_starts_lists_and_cancels_tasks(pipe: PipeInput) -> None:
    kit = RoomKit()
    await _tasks_room(kit)
    console, running = await _console(kit, pipe)
    assert "/task @dev" in _room(console.view)

    pipe.send_text("/task @dev read the tax engine's source\r")
    await _until(lambda: "started: @helper" in _room(console.view))
    await _until(lambda: "1 task running" in _panel(console.view))
    (task_id,) = console.view.working
    short = task_id.removeprefix("task-")[:6]

    pipe.send_text("/tasks\r")
    await _until(lambda: f"{short}  @dev" in _room(console.view))
    assert "running" in _room(console.view)

    pipe.send_text(f"/cancel {short}\r")
    await _until(lambda: f"task {short} cancelled" in _room(console.view))
    await _until(lambda: "task running" not in _panel(console.view))
    console.exit()
    await asyncio.wait_for(running, 5)
    await kit.close()


async def test_task_commands_are_off_without_background_work(pipe: PipeInput) -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    dev = Agent("dev", provider=MockAIProvider(ai_responses=[AIResponse(content="ok")]))
    await kit.create_room(room_id="r1", orchestration=Discussion([dev]))
    await kit.attach_channel("r1", "ops")
    console, running = await _console(kit, pipe)

    assert "/task" not in _room(console.view)
    pipe.send_text("/task @dev something\r")
    await _until(lambda: "unknown command /task" in _room(console.view))
    console.exit()
    await asyncio.wait_for(running, 5)
    await kit.close()


def test_another_stream_and_a_posted_result_leave_the_live_text_alone() -> None:
    view = DiscussionView(CARDS, you="ops")
    view.live_start("dev", "s1")
    view.live_text("Reading the ", "s1")
    view.live_text("ignored", "s2")
    view.agent("dev", [], "Result of the task “x”:\n\nFindings.", by="dev-code")
    view.live_end("s2")

    assert view.live is not None and view.live.text == "Reading the "


def test_a_turn_that_ends_clears_a_stream_cut_short() -> None:
    view = DiscussionView(CARDS, you="ops")
    view.live_start("dev", "s1")
    view.live_text("half an answ", "s1")
    view.queue_changed(_turn(SpeakQueueChange.TURN_ENDED, "dev"))
    assert view.live is None


def _turn(change: SpeakQueueChange, *agents: str) -> SpeakQueueEvent:
    return SpeakQueueEvent(
        room_id="r1",
        queue=SpeakQueue(
            speaking=None, queue=(), listening=frozenset(), asked=(), waiting=False, over=False
        ),
        change=change,
        channel_ids=agents,
    )


def test_an_agent_card_never_reads_with_two_at_signs() -> None:
    view = DiscussionView(CARDS, you="ops")
    view.queue_changed(_asked(("sre", "@sms1")))
    assert "asked @sms1" in _panel(view)
    assert "@@" not in _panel(view)


async def test_a_log_file_that_cannot_be_opened_leaves_nothing_behind(
    pipe: PipeInput, tmp_path: Any
) -> None:
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("ops"))
    dev = Agent("dev", provider=MockAIProvider(ai_responses=[AIResponse(content="ok")]))
    await kit.create_room(room_id="r1", orchestration=Discussion([dev]))
    await kit.attach_channel("r1", "ops")
    root = logging.getLogger()
    before = list(root.handlers)
    console = DiscussionConsole(kit, "r1", channel_id="ops", log_file=tmp_path / "no" / "x.log")

    with pytest.raises(OSError):
        await console.run()

    assert not any(
        h.name.startswith("discussion_console_") for h in kit.hook_engine._room_hooks.get("r1", [])
    )
    assert kit.channels["ops"].connection_count == 0  # type: ignore[attr-defined]
    assert root.handlers == before
    await kit.close()
