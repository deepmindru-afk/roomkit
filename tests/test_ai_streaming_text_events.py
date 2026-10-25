"""An agent's answer text, published live as TEXT_DELTA ephemerals (RFC §6.4, RMK-951).

An observer that is not a transport of the room (a debugger, a dashboard, the
code that delegated a task) reads the answer as it is written, coalesced on a
window, in the room the agent speaks in, a delegated task's child room
included, once the host turns it on with ``publish_text_deltas=True``.
BEFORE_BROADCAST does not cover that stream, so it is off by default.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    SMSChannel,
    TextContent,
)
from roomkit.channels.agent import Agent
from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    AITool,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.providers.sms.mock import MockSMSProvider

WORDS = [f"word{i} " for i in range(100)]
ANSWER = "".join(WORDS)


class _Rounds(AIProvider):
    """Streams scripted rounds: each a list of text tokens, then an optional
    tool call ending the round."""

    def __init__(self, rounds: list[tuple[list[str], StreamToolCall | None]]) -> None:
        self._rounds = list(rounds)

    @property
    def model_name(self) -> str:
        return "rounds"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        tokens, _ = self._rounds[0]
        return AIResponse(content="".join(tokens), finish_reason="stop")

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        for token in self._rounds[0][0]:
            yield token

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        tokens, call = self._rounds.pop(0)
        for token in tokens:
            yield StreamTextDelta(text=token)
        if call is not None:
            yield call
        yield StreamDone(finish_reason="tool_calls" if call else "stop", usage={})


class _Observer:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, event: Any) -> None:
        self.events.append((event.type.value, event.data or {}))

    def texts(self) -> list[dict[str, Any]]:
        return [data for kind, data in self.events if kind == "text_delta"]


@pytest.fixture
async def kit():
    kit = RoomKit()
    yield kit
    await kit.close()


@pytest.mark.parametrize(
    ("coalesce_ms", "publishes"),
    # 690 characters: one publish per token, or one per 256-character window.
    [(0, len(WORDS)), (80, 3)],
    ids=["window off: one per token", "default window: by size"],
)
async def test_a_streamed_answer_reaches_a_room_subscriber_as_its_text(
    kit, coalesce_ms, publishes
):
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    kit.register_channel(
        Agent(
            "assistant",
            provider=_Rounds([(WORDS, None)]),
            publish_text_deltas=True,
            text_coalesce_ms=coalesce_ms,
        )
    )
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms", metadata={"phone_number": "+15550100001"})
    await kit.attach_channel("r", "assistant", category=ChannelCategory.INTELLIGENCE)
    observer = _Observer()
    await kit.realtime.subscribe_to_room("r", observer)

    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="+15550100001", content=TextContent(body="go")),
        room_id="r",
    )

    texts = observer.texts()
    assert "".join(data["text"] for data in texts) == ANSWER
    assert len(texts) == publishes
    assert {data["channel_id"] for data in texts} == {"assistant"}


async def test_a_delegated_turns_text_reaches_its_child_room_while_it_is_written(kit):
    kit.register_channel(
        Agent("worker", provider=_Rounds([(WORDS, None)]), publish_text_deltas=True)
    )
    observer = _Observer()
    order: list[str] = []

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def watch(event, ctx) -> None:
        async def seen(ephemeral: Any) -> None:
            await observer(ephemeral)
            order.append("delta")

        await kit.realtime.subscribe_to_room(event.metadata["child_room_id"], seen)

    room = await kit.create_room()
    result = await (await kit.delegate(room.id, "worker", "go", wait=True)).wait()
    order.append("returned")

    assert "".join(data["text"] for data in observer.texts()) == result.output == ANSWER
    assert order[-1] == "returned" and "delta" in order


async def test_text_before_a_tool_call_is_published_before_it_with_its_round(kit):
    call = StreamToolCall(id="c1", name="lookup", arguments={"order": "1042"})
    provider = _Rounds([(["Let me check. "], call), (["It shipped today."], None)])

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        return "shipped"

    kit.register_channel(
        Agent(
            "worker",
            provider=provider,
            tools=[AITool(name="lookup", description="Look up an order", parameters={})],
            tool_handler=lookup,
            publish_text_deltas=True,
        )
    )
    observer = _Observer()

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def watch(event, ctx) -> None:
        await kit.realtime.subscribe_to_room(event.metadata["child_room_id"], observer)

    room = await kit.create_room()
    await (await kit.delegate(room.id, "worker", "where is my order?", wait=True)).wait()

    kinds = [kind for kind, _ in observer.events]
    assert kinds.index("text_delta") < kinds.index("tool_call_start")
    assert [(data["round"], data["text"]) for data in observer.texts()] == [
        (0, "Let me check. "),
        (1, "It shipped today."),
    ]


SECRET = "Your card 4111 1111 1111 1111 is on file."
_MODERATIONS = {
    "block": (lambda event: HookResult.block("pii"), []),
    "redact": (
        lambda event: HookResult.modify(
            event.model_copy(update={"content": TextContent(body="[redacted]")})
        ),
        ["[redacted]"],
    ),
}


@pytest.mark.parametrize("moderation", sorted(_MODERATIONS))
@pytest.mark.parametrize(
    ("publish", "read"), [(None, ""), (True, SECRET)], ids=["default", "turned on"]
)
async def test_before_broadcast_does_not_cover_the_text_stream(
    kit, moderation: str, publish: bool | None, read: str
) -> None:
    """A hook that blocks or redacts the answer judges the stored message,
    never the TEXT_DELTA stream: turned on, the stream hands a room
    subscriber the text the room never got. So it is off unless the host
    asks, and a host that filters or redacts answers leaves it off."""
    judge, sent = _MODERATIONS[moderation]
    sms = MockSMSProvider()
    kit.register_channel(SMSChannel("sms", provider=sms))
    options = {} if publish is None else {"publish_text_deltas": publish}
    kit.register_channel(Agent("assistant", provider=_Rounds([([SECRET], None)]), **options))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms", metadata={"phone_number": "+15550100001"})
    await kit.attach_channel("r", "assistant", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def moderate(event, ctx) -> HookResult:
        if event.source.channel_id == "assistant":
            return judge(event)
        return HookResult.allow()

    observer = _Observer()
    await kit.realtime.subscribe_to_room("r", observer)

    await kit.process_inbound(
        InboundMessage(
            channel_id="sms", sender_id="+15550100001", content=TextContent(body="my card?")
        ),
        room_id="r",
    )

    assert [item["event"].content.body for item in sms.sent] == sent
    assert "".join(data["text"] for data in observer.texts()) == read
