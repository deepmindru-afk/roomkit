"""A live agent's turns run on the kit's own executor, whatever other executor
is served on the kit's turn backend: the agent the kit holds answers, and the
kit's hooks fire for it (RFC §10.6.6, §10.6.7, RMK-837).

The kit's backend is in memory, or across Redis in the suite's Redis mode
(tests/redis_turn_mode.py), so each test runs on both."""

from __future__ import annotations

from collections.abc import Callable

from roomkit import (
    AIChannel,
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    TextContent,
    TurnExecutor,
    WebSocketChannel,
)
from roomkit.models.delivery import InboundMessage
from roomkit.providers.ai.mock import MockAIProvider


async def _answers(
    build: Callable[[str], AIChannel | None], *, room_hook: str | None = None
) -> tuple[list[str], list[str]]:
    """What the room's agent said to one message, and the channels the kit's
    BEFORE_AI_GENERATION hook saw, with a host executor built on *build*
    served on the kit's backend before its first turn; *room_hook*, defined on
    the host's executor alone, enabled for the room."""
    kit = RoomKit()
    host = TurnExecutor(build)
    if room_hook is not None:

        @host.room_hook(room_hook, HookTrigger.ON_AI_RESPONSE, HookExecution.ASYNC)
        async def audit(event, ctx):
            pass

    await kit._turns.backend.serve(host)
    kit.register_channel(WebSocketChannel("ws"))
    kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["from the kit"])))
    seen: list[str] = []

    @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
    async def observe(event, ctx):
        seen.append(event.channel_id)
        return HookResult.allow()

    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "ws")
    await kit.attach_channel("r", "ai", category=ChannelCategory.INTELLIGENCE)
    if room_hook is not None:
        kit.enable_room_hook("r", room_hook)
    await kit.process_inbound(
        InboundMessage(channel_id="ws", sender_id="u", content=TextContent(body="hi"))
    )
    events = await kit.store.list_events("r")
    await kit.close()
    said = [e.content.body for e in events if e.source.channel_id == "ai"]
    return said, seen


async def test_a_host_executor_served_first_does_not_take_a_live_agent_s_turn() -> None:
    """The host's executor declares every channel and was served first: it
    used to take the turn, its own agent answering and the kit's hooks never
    firing."""
    providers: list[MockAIProvider] = []

    def build(channel_id: str) -> AIChannel:
        providers.append(MockAIProvider(responses=["from the host"]))
        return AIChannel(channel_id, provider=providers[-1])

    said, seen = await _answers(build)

    assert said == ["from the kit"]
    assert seen == ["ai"]
    assert all(not provider.calls for provider in providers)


async def test_a_live_agent_answers_when_the_host_has_no_channel_for_it() -> None:
    """The host's factory builds nothing for the kit's agent: its turn used to
    fail on the host executor, which declared every channel."""

    said, seen = await _answers(lambda channel_id: None)

    assert said == ["from the kit"]
    assert seen == ["ai"]


async def test_a_live_agent_s_turn_never_moves_to_a_host_for_a_room_hook() -> None:
    """The room enables a hook only the host's executor defines: the live
    agent's turn finds no executor, as the kit warns when the hook is enabled,
    rather than run on the host's copy, which defines it."""
    providers: list[MockAIProvider] = []

    def build(channel_id: str) -> AIChannel:
        providers.append(MockAIProvider(responses=["from the host"]))
        return AIChannel(channel_id, provider=providers[-1])

    said, seen = await _answers(build, room_hook="audit")

    assert said == []
    assert seen == []
    assert all(not provider.calls for provider in providers)
