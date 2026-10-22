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
from roomkit.models.enums import ChannelCategory
from roomkit.orchestration._worker_run import WorkerStatus, run_worker
from roomkit.orchestration.strategies.discussion import (
    SpeakQueue,
    SpeakQueueChange,
    SpeakQueueEvent,
)
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.handback import CALLER_HANDS_BACK
from roomkit.tasks.status import post_task_progress
from roomkit.tools.context import tool_turn_context
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


async def _tasks_room(kit: RoomKit, dev_id: str = "dev", name: str | None = None) -> None:
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(Agent("helper", provider=_Slow("Findings.", 5.0)))
    reply = MockAIProvider(ai_responses=[AIResponse(content="ok")])
    dev = Agent(dev_id, provider=reply, name=name)
    sre = Agent("sre", provider=MockAIProvider(ai_responses=[AIResponse(content="ok")]))
    kit.setup_background(dev, worker="helper")
    await kit.create_room(room_id="r1", orchestration=Discussion([dev, sre]))
    await kit.attach_channel("r1", "ops")


@pytest.mark.parametrize(
    ("dev_id", "name", "handle"),
    [("dev", None, "dev"), ("agent:8e41f0d2", "Dev", "Dev")],
    ids=["channel id", "handle"],
)
async def test_the_person_starts_lists_and_cancels_tasks(
    pipe: PipeInput, dev_id: str, name: str | None, handle: str
) -> None:
    """With a handle (RMK-764), task commands are on, ``@dev`` reaches the
    agent, and the console names it by its handle."""
    kit = RoomKit()
    await _tasks_room(kit, dev_id, name)
    console, running = await _console(kit, pipe)
    assert f"/task @{handle}" in _room(console.view)

    pipe.send_text("/task @dev read the tax engine's source\r")
    await _until(lambda: "started: @helper" in _room(console.view))
    await _until(lambda: "1 task running" in _panel(console.view))
    (task_id,) = console.view.working
    short = task_id.removeprefix("task-")[:6]

    pipe.send_text("/tasks\r")
    await _until(lambda: f"{short}  @{handle}" in _room(console.view))
    assert "running" in _room(console.view)

    pipe.send_text(f"/cancel {short}\r")
    await _until(lambda: f"task {short} cancelled" in _room(console.view))
    await _until(lambda: "task running" not in _panel(console.view))
    console.exit()
    await asyncio.wait_for(running, 5)
    await kit.close()


async def _inline_run(kit: RoomKit, door: str) -> asyncio.Future[Any]:
    """An inline run of the helper's in r1 no cancel reaches: a delegation
    inside its turn, or a strategy's worker run, which posts its progress so
    that the room lists it by its task."""
    if door == "delegation":
        return asyncio.ensure_future(kit.delegate("r1", "helper", "Read the source.", wait=True))
    run = asyncio.ensure_future(
        run_worker(
            kit,
            "r1",
            "helper",
            "Read the source.",
            timeout=None,
            status=WorkerStatus({"room_id": "r1", "strategy": "sequential"}),
        )
    )
    await _until(lambda: len(kit._room_tasks.running) == 1)
    child = next(r.id for r in await kit.store.list_rooms() if r.id.startswith("r1::task-"))
    with tool_turn_context(room_id=child):
        assert await post_task_progress(kit, "2 of 9")
    return run


@pytest.mark.parametrize(
    ("door", "why"),
    [
        ("delegation", "it runs inside the turn that asked for it"),
        ("strategy", "the strategy that runs it follows it"),
    ],
)
async def test_the_console_says_why_an_inline_run_is_not_cancelled(
    pipe: PipeInput, door: str, why: str
) -> None:
    """RMK-829: ``/cancel`` on an inline run no cancel reaches says it is
    still running and why, as ``cancel_task`` does, never that it had ended."""
    kit = RoomKit()
    await _tasks_room(kit)
    console, running = await _console(kit, pipe)
    inline = await _inline_run(kit, door)
    await _until(lambda: len(kit._room_tasks.running) == 1)
    (task_id,) = kit._room_tasks.running
    short = task_id.removeprefix("task-")[:6]
    pipe.send_text("/tasks\r")
    await _until(lambda: f"{short}  @helper" in _room(console.view))

    pipe.send_text(f"/cancel {short}\r")
    await _until(lambda: f"task {short} is still running: {why}" in _room(console.view))

    assert not inline.done()
    inline.cancel()
    await asyncio.gather(inline, return_exceptions=True)
    console.exit()
    await asyncio.wait_for(running, 5)
    await kit.close()


async def test_the_console_cancels_no_task_a_line_names_for_another_room(
    pipe: PipeInput,
) -> None:
    """RMK-829: a line posted for r1 naming another room's running task: the
    console's ``/cancel`` in r1 cancels nothing, whatever the line says."""
    kit = RoomKit()
    await _tasks_room(kit)
    await kit.create_room(room_id="other")
    await kit.attach_channel("other", "dev", category=ChannelCategory.INTELLIGENCE)
    console, running = await _console(kit, pipe)
    theirs = await kit.delegate("other", "helper", "Read the source.", notify="dev")
    kit.status_bus.post(
        "helper", "task", "pending", metadata={"room_id": "r1", "task_id": theirs.id}
    )
    await asyncio.sleep(0.05)
    short = theirs.id.removeprefix("task-")[:6]

    pipe.send_text(f"/cancel {short}\r")
    await _until(lambda: f"task {short} is not one this kit runs" in _room(console.view))

    assert theirs.result is None
    await kit.cancel_task(theirs.id)
    console.exit()
    await asyncio.wait_for(running, 5)
    await kit.close()


async def test_the_console_reads_a_task_that_ended_while_its_cancel_waited(
    pipe: PipeInput,
) -> None:
    """RMK-829: a strategy's background task ends while ``/cancel`` waits on
    it: the console says it had ended, never that its strategy follows it."""
    kit = RoomKit()
    await _tasks_room(kit)
    console, running = await _console(kit, pipe)
    task = await kit.delegate(
        "r1", "helper", "Read the source.", notify=CALLER_HANDS_BACK, post_status=False
    )
    with tool_turn_context(room_id=task.child_room_id):
        assert await post_task_progress(kit, "2 of 9")
    short = task.id.removeprefix("task-")[:6]

    async def ends_meanwhile(task_id: str, channel_id: str | None, **_: object) -> bool:
        kit._room_tasks.end(task_id, "completed")
        return False

    kit._cancel_task_for = ends_meanwhile  # type: ignore[method-assign]
    pipe.send_text(f"/cancel {short}\r")
    await _until(lambda: f"task {short} had ended" in _room(console.view))

    console.exit()
    await asyncio.wait_for(running, 5)
    await kit.close()


async def test_the_console_says_a_task_its_line_lists_ended_had_ended(
    pipe: PipeInput,
) -> None:
    """RMK-829: a line of the room listing a task ended, one this kit never
    ran included: ``/cancel`` says it had ended, as ``/tasks`` lists it."""
    kit = RoomKit()
    await _tasks_room(kit)
    console, running = await _console(kit, pipe)
    kit.status_bus.post(
        "helper",
        "task",
        "completed",
        metadata={"room_id": "r1", "task_id": "task-a1b2c3d4", "task_status": "completed"},
    )
    await asyncio.sleep(0.05)

    pipe.send_text("/cancel a1b2c3\r")
    await _until(lambda: "task a1b2c3 had ended" in _room(console.view))

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

    for engine in (kit.hook_engine, kit.turn_executor.hooks):
        assert not any(
            h.name.startswith("discussion_console_") for h in engine._room_hooks.get("r1", [])
        )
    assert kit.channels["ops"].connection_count == 0  # type: ignore[attr-defined]
    assert root.handlers == before
    await kit.close()
