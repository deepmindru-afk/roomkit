"""A transport shared into a delegated room is told the agent's answers (RMK-360).

RFC §23.3 steps 4 to 6: the task description and a result tool's re-prompt
are the delegating side's instruction to the worker and reach no shared
transport; the agent's answers reach it through the room's own gates
(``BEFORE_BROADCAST``, the agent's right to write, the delivery lane), so the
shared binding's access decides what it receives, as in any room. Without a
shared transport, the child room keeps its trace as before, past no hook.
"""

from __future__ import annotations

import pytest

from roomkit import ChannelCategory, HookResult, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.models.context import RoomContext
from roomkit.models.enums import Access
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.store_filter import EventFilter
from roomkit.orchestration.result import SUBMIT_RESULT
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel

TASK = "Summarise the ticket and email the customer."


async def _kit(
    *answers: str,
    streaming: bool = False,
    access: Access = Access.READ_WRITE,
    muted: bool = False,
) -> tuple[RoomKit, SimpleChannel]:
    kit = RoomKit()
    kit.register_channel(
        Agent(
            "worker",
            provider=MockAIProvider(responses=list(answers), streaming=streaming),
            role="Writer",
            description="Writes summaries",
        )
    )
    email = SimpleChannel("email-out")
    kit.register_channel(email)
    await kit.create_room(room_id="parent")
    await kit.attach_channel("parent", "email-out", access=access, muted=muted)
    return kit, email


def _bodies(channel: SimpleChannel) -> list[str]:
    return [e.content.body for e in channel.delivered if isinstance(e.content, TextContent)]


class TestASharedTransportIsToldTheAnswer:
    @pytest.mark.parametrize("streaming", [False, True])
    async def test_it_receives_the_answer_and_never_the_task(self, streaming: bool) -> None:
        kit, email = await _kit("Your ticket is solved.", streaming=streaming)

        task = await kit.delegate(
            "parent", "worker", TASK, share_channels=["email-out"], wait=True
        )

        assert _bodies(email) == ["Your ticket is solved."]
        assert task.result is not None
        assert task.result.output == "Your ticket is solved."
        await kit.close()

    async def test_a_result_tool_reprompt_never_reaches_it(self) -> None:
        kit, email = await _kit("First try.", "Second try.")

        await kit.delegate(
            "parent",
            "worker",
            TASK,
            share_channels=["email-out"],
            wait=True,
            require_structured_result=True,
            max_result_retries=1,
        )

        # The two answers are the agent's; the task and the re-prompt are not.
        assert _bodies(email) == ["First try.", "Second try."]
        assert TASK not in _bodies(email)
        assert SUBMIT_RESULT.reminder not in _bodies(email)
        await kit.close()

    async def test_the_answer_crosses_the_rooms_hooks(self) -> None:
        kit, email = await _kit("Call me at 555-0100.")

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def redact(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if isinstance(event.content, TextContent) and "555-0100" in event.content.body:
                body = event.content.body.replace("555-0100", "[phone]")
                return HookResult.modify(
                    event.model_copy(update={"content": TextContent(body=body)})
                )
            return HookResult.allow()

        task = await kit.delegate(
            "parent", "worker", TASK, share_channels=["email-out"], wait=True
        )

        assert _bodies(email) == ["Call me at [phone]."]
        # The rewrite holds for the task result too (RFC §23.3 step 6).
        assert task.result is not None
        assert task.result.output == "Call me at [phone]."
        await kit.close()

    @pytest.mark.parametrize(
        ("access", "muted", "told"),
        [
            (Access.READ_ONLY, False, True),
            (Access.READ_WRITE, True, True),
            (Access.WRITE_ONLY, False, False),
        ],
    )
    async def test_its_binding_decides_what_it_reads(
        self, access: Access, muted: bool, told: bool
    ) -> None:
        """RFC §7.5 rule 1: a binding that may read is told, muted or not; one
        that may not is told nothing. The parent's permissions ride along."""
        kit, email = await _kit("Done.", access=access, muted=muted)

        task = await kit.delegate(
            "parent", "worker", TASK, share_channels=["email-out"], wait=True
        )

        assert _bodies(email) == (["Done."] if told else [])
        assert task.result is not None
        assert task.result.output == "Done."
        await kit.close()


class TestWithoutASharedTransport:
    async def test_the_trace_crosses_no_hook(self) -> None:
        kit, email = await _kit("Done.")
        hooked: list[str] = []

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def record(event: RoomEvent, ctx: RoomContext) -> HookResult:
            hooked.append(event.room_id)
            return HookResult.allow()

        task = await kit.delegate("parent", "worker", TASK, wait=True)

        assert task.result is not None
        assert task.result.output == "Done."
        assert email.delivered == []
        assert not [room for room in hooked if room == task.child_room_id]
        rows = await kit.store.list_events(
            task.child_room_id, event_filter=EventFilter(include_blocked=True)
        )
        assert [r.content.body for r in rows if isinstance(r.content, TextContent)] == [
            TASK,
            "Done.",
        ]
        await kit.close()

    async def test_a_shared_agent_still_reads_the_task(self) -> None:
        """The task is scoped to the room's agents, not hidden from them."""
        kit = RoomKit()
        kit.register_channel(
            Agent(
                "worker", provider=MockAIProvider(responses=["Done."]), role="r", description="d"
            )
        )
        observer = Agent(
            "observer", provider=MockAIProvider(responses=["Noted."]), role="r", description="d"
        )
        kit.register_channel(observer)
        await kit.create_room(room_id="parent")
        await kit.attach_channel("parent", "observer", category=ChannelCategory.INTELLIGENCE)

        await kit.delegate("parent", "worker", TASK, share_channels=["observer"], wait=True)

        assert any(TASK in str(call.messages) for call in observer._provider.calls)
        await kit.close()
