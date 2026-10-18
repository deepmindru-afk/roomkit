"""Ask a person from a speech-to-speech session with human_input_handler=.

    uv run python examples/realtime_human_input.py

A realtime voice model calls ``ask_operator`` when the caller asks for
something only a person can grant. The channel serves the tool itself, before
its ``tool_handler``, as an ``AIChannel`` does:

- each request fires ``ON_USER_INPUT_REQUIRED``, its ``channel_type`` naming
  the door it was asked on (``realtime_voice``); a BLOCK would reject it;
- the handler's own ``timeout`` bounds the call, not the channel's default
  call bound, which is set very short here to show it does not apply: the
  operator answers after it has long expired;
- the operator's answer is the call's result, sent to the model.

Runs on the mock realtime provider, so it needs no API key: the model's tool
call is simulated.
"""

from __future__ import annotations

import asyncio
from typing import Any

from shared import setup_logging

from roomkit import (
    HookExecution,
    HookResult,
    HookTrigger,
    HumanInputToolHandler,
    RealtimeVoiceChannel,
    RoomKit,
)
from roomkit.providers.ai.base import AITool
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

logger = setup_logging("realtime_human_input")

OPERATOR_ANSWERS_AFTER = 0.5
"""Seconds the operator takes, well past the channel's default call bound."""

ASK_OPERATOR = AITool(
    name="ask_operator",
    description="Ask the human operator to approve something the caller requested.",
    parameters={
        "type": "object",
        "properties": {"request": {"type": "string"}},
        "required": ["request"],
    },
)


async def main() -> None:
    human = HumanInputToolHandler({"ask_operator"}, timeout=30, tool_definitions=[ASK_OPERATOR])
    provider = MockRealtimeProvider()
    voice = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=MockRealtimeTransport(),
        system_prompt="You are a front desk agent. Ask the operator before granting refunds.",
        human_input_handler=human,
        tool_timeout_seconds=0.1,  # the default bound; ask_operator keeps its own
    )
    kit = RoomKit()
    kit.register_channel(voice)
    await kit.create_room(room_id="front-desk")
    await kit.attach_channel("front-desk", "voice")

    @kit.hook(HookTrigger.ON_USER_INPUT_REQUIRED, execution=HookExecution.SYNC)
    async def notify_operator(event: Any, ctx: Any) -> HookResult:
        logger.info(
            "operator asked on a %s door: %s %s (caller %s)",
            event.channel_type,
            event.tool_name,
            event.arguments,
            event.actor_id,
        )

        async def operator_answers() -> None:
            await asyncio.sleep(OPERATOR_ANSWERS_AFTER)
            human.handler.resolve(event.pending_id, '{"approved": true, "note": "one time only"}')

        # The notification never gates the answer: the operator answers on
        # their own time, from a dashboard in a real deployment.
        asyncio.get_running_loop().create_task(operator_answers())
        return HookResult.allow()

    session = await voice.start_session("front-desk", "caller-1", "ws")
    await provider.simulate_tool_call(
        session, "call-1", "ask_operator", {"request": "Refund my last booking"}
    )
    for _ in range(100):
        if provider.tool_results:
            break
        await asyncio.sleep(0.05)

    for _, call_id, result in provider.tool_results:
        logger.info("the model reads the result of %s: %s", call_id, result)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
