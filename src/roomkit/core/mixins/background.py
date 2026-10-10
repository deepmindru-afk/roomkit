"""BackgroundMixin — long work started as a task (RFC §23.4, §19.7.5 rule 19)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.channels._tool_registry import orchestration_tool
from roomkit.orchestration._call_room import in_call_room
from roomkit.tasks.background import (
    BACKGROUND_TOOL,
    BACKGROUND_TOOL_DEF,
    BackgroundSetup,
    background_tool_server,
    start_background_task,
)

if TYPE_CHECKING:
    from roomkit.channels.ai import AIChannel
    from roomkit.tasks.models import DelegatedTask


class BackgroundMixin:
    """``setup_background`` and ``start_task``: an agent's long work runs as a
    task, off its turn."""

    _background_setups: dict[str, BackgroundSetup]

    def setup_background(
        self, agent: AIChannel, *, worker: str | None = None, hand_back: str = "relay"
    ) -> None:
        """Give *agent* the ``work_in_background`` tool (RFC §23.4): it starts
        long work as a task and ends its turn, the work running apart.

        Args:
            agent: The AI channel given the tool, in every room it serves.
            worker: The channel that does its tasks (another agent, an
                external agent over ACP); the agent itself when None.
            hand_back: How a task's result reaches the room: ``"relay"``, the
                agent says it in a turn of its own, or ``"post"``, it is
                posted as it is, as the agent's message.

        Raises:
            ValueError: An unknown *hand_back*.
            RuntimeError: The tool is already set up on the agent.
        """
        setup = BackgroundSetup(worker=worker, hand_back=hand_back)
        if agent._registry.lookup(BACKGROUND_TOOL, None) is not None:
            msg = f"setup_background() already called for channel '{agent.channel_id}'"
            raise RuntimeError(msg)
        serve = background_tool_server(self, agent.channel_id)  # ty: ignore[invalid-argument-type]
        agent._registry.register(
            orchestration_tool(BACKGROUND_TOOL_DEF, in_call_room(BACKGROUND_TOOL, serve)),
            owner=self,
        )
        self._background_setups[agent.channel_id] = setup

    async def start_task(
        self, room_id: str, agent_id: str, task: str, *, hand_back: str | None = None
    ) -> DelegatedTask:
        """Start *task* for *agent_id*, an agent of *room_id*, as its
        ``work_in_background`` call would (RFC §23.4): on its worker, handed
        the room's last messages, its result relayed or posted back to it.
        The host's door, and a person's through what the host gives them; an
        agent with no background set up does the work itself and has its
        result relayed.

        Args:
            room_id: The room the task is for.
            agent_id: The agent the task is for: an intelligence channel
                attached to the room, an external agent included.
            task: What to do and report.
            hand_back: ``"relay"`` or ``"post"``; the agent's setup's when None.

        Raises:
            ValueError: *agent_id* is no agent of the room, *task* is blank,
                or *hand_back* is unknown.
        """
        setup = self._background_setups.get(agent_id) or BackgroundSetup()
        return await start_background_task(
            self,  # ty: ignore[invalid-argument-type]
            room_id,
            agent_id,
            task,
            setup=setup,
            hand_back=hand_back,
        )
