"""A Loop whose producer's task failed says so (RMK-435, RFC §19.7.4, §23.3).

The sync Loop no longer publishes an empty producer message: with no output
at all the turn has no answer and the caller reads why; with an earlier
output, that output goes out, not approved, with why the loop stopped. The
async Loop's delivered text names the real reason. The voice
``delegate_workers`` of a Supervisor waits on its workers like its twins, and
a stream read for a supervisor's task is no answer when its turn was cut.
"""

from __future__ import annotations

from typing import Any

from roomkit import RoomKit, TaskCutShortError
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.channel import ChannelOutput
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventType
from roomkit.models.event import TextContent
from roomkit.models.streaming import LoopEndMarker
from roomkit.orchestration.strategies.loop import Loop, _async_loop_and_deliver
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.orchestration.strategies.supervisor.results import _extract_output_text
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_framework import SimpleChannel

LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})
LOOPING = AIResponse(
    content="Still checking.",
    finish_reason="tool_calls",
    tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
)


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


def _producer(responses: list[AIResponse]) -> Agent:
    return Agent(
        "producer",
        provider=MockAIProvider(ai_responses=responses, streaming=True),
        tools=[LOOKUP],
        tool_handler=_found,
        tool_search=False,
        max_tool_rounds=1,
    )


async def _sync_loop(producer: Agent, reviews: list[str]) -> tuple[RoomKit, Any]:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(producer)
    reviewer = Agent("reviewer", provider=MockAIProvider(responses=reviews))
    await kit.create_room(
        room_id="r", orchestration=Loop(agent=producer, reviewer=reviewer, max_iterations=3)
    )
    await kit.attach_channel("r", "sms")
    result = await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Write it."))
    )
    return kit, result


async def _producer_messages(kit: RoomKit) -> list[tuple[str | None, dict[str, Any]]]:
    return [
        (e.content.body if isinstance(e.content, TextContent) else None, e.metadata)
        for e in await kit.get_timeline("r", limit=50)
        if e.source.channel_id == "producer" and e.type == EventType.MESSAGE
    ]


async def test_a_producer_cut_before_any_output_gives_no_answer_and_its_reason() -> None:
    kit, result = await _sync_loop(_producer([LOOPING] * 10), ["APPROVED"])

    assert await _producer_messages(kit) == []
    assert isinstance(result.error, TaskCutShortError)
    assert result.error.reason == "max_rounds"
    await kit.close()


async def test_a_producer_cut_after_an_output_keeps_it_not_approved() -> None:
    first = AIResponse(content="Draft one.")
    kit, result = await _sync_loop(_producer([first, *[LOOPING] * 10]), ["Needs work."] * 3)

    [(body, metadata)] = await _producer_messages(kit)
    assert body == "Draft one."
    assert (metadata["approved"], metadata["stopped"]) == (False, "producer_failed")
    assert result.error is None
    await kit.close()


async def test_the_async_loop_names_the_producers_failure() -> None:
    producer = _producer([LOOPING] * 10)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(producer)
    reviewer = Agent("reviewer", provider=MockAIProvider(responses=["APPROVED"]))
    kit.register_channel(reviewer)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    delivered: list[str] = []
    real_deliver = kit.deliver

    async def spy(room_id: str, content: Any, **kw: Any) -> Any:
        delivered.append(str(content))
        return await real_deliver(room_id, content, **kw)

    kit.deliver = spy  # type: ignore[method-assign]
    await _async_loop_and_deliver(
        kit=kit,
        room_id="r",
        producer=producer,
        reviewers=[reviewer],
        strategy=None,
        task_desc="Write it.",
        max_iterations=3,
        on_done=lambda: None,
    )

    [text] = delivered
    assert "the producer's task failed" in text
    assert "max iterations reached" not in text
    await kit.close()


async def test_the_voice_delegate_workers_waits_on_its_workers() -> None:
    kit = RoomKit()
    voice = RealtimeVoiceChannel(
        "voice", provider=MockRealtimeProvider(), transport=MockRealtimeTransport()
    )
    kit.register_channel(voice)
    worker = Agent("worker", provider=MockAIProvider())
    supervisor = Agent("sup", provider=MockAIProvider())
    kit.register_channel(worker)
    kit.register_channel(supervisor)
    await kit.create_room(
        room_id="r",
        orchestration=Supervisor(
            supervisor=supervisor,
            workers=[worker],
            strategy="parallel",
            auto_delegate=True,
            async_delivery=True,
        ),
    )

    assert voice._call_timeout("delegate_workers", "r") is None
    await kit.close()


async def test_a_cut_streamed_task_text_is_no_answer() -> None:
    async def stream() -> Any:
        yield "Still checking."
        yield LoopEndMarker(reason="max_rounds", rounds=1, usage={})

    assert await _extract_output_text(ChannelOutput(response_stream=stream())) == ""
