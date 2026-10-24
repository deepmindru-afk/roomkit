"""An application's error crosses the turn backend as its own class.

Every turn of an AI channel runs through the kit's turn backend (RFC §10.6),
in memory by default, as serialized messages: an error the turn raised is
rebuilt on the kit's side (RFC §10.6.8). Here the model's stream fails with
the application's own ``QuotaExceededError``, a ``ProviderError`` carrying
its code and the delay before a retry:

1. A kit that declares nothing reads it as a ``RemoteTurnError``. It keeps
   the class's name, its attributes and its ancestor: it is still a
   ``ProviderError``, so the kit logs a provider line, not a traceback.
2. A kit built with ``error_classes=(QuotaExceededError,)`` reads the
   application's class itself, as a turn run in the same process would raise
   it. An executor in another process declares the same classes on its own
   ``TurnExecutor(error_classes=...)``.

The model is a scripted mock, so the example runs without keys.

Run with:
    uv run python examples/turn_error_classes.py
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    AIChannel,
    ChannelCategory,
    HookExecution,
    HookTrigger,
    InboundMessage,
    RemoteTurnError,
    RoomContext,
    RoomEvent,
    RoomKit,
    SMSChannel,
    TextContent,
)
from roomkit.providers.ai import AIContext, MockAIProvider, ProviderError, StreamEvent
from roomkit.providers.sms import MockSMSProvider

logger = setup_logging("example.turn_error_classes")


class QuotaExceededError(ProviderError):
    """The application's own reading of a provider's refusal."""

    def __init__(self, message: str, *, code: str, retry_after: float) -> None:
        super().__init__(message, provider="vendor", status_code=429, retryable=True)
        self.code = code
        self.retry_after = retry_after


class QuotaModel(MockAIProvider):
    """A model whose stream fails with the application's error."""

    def __init__(self) -> None:
        super().__init__(streaming=True)

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        raise QuotaExceededError("Monthly quota reached", code="quota_exceeded", retry_after=30.0)
        yield  # pragma: no cover - keeps this an async generator


async def ask(kit: RoomKit) -> tuple[BaseException | None, list[str]]:
    """One message to a room whose agent's model fails: the error the kit read,
    and the ``error_type`` its ON_ERROR hooks were given."""
    reported: list[str] = []

    @kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC)
    async def on_error(event: RoomEvent, ctx: RoomContext) -> None:
        reported.append(event.metadata["error_type"])

    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    kit.register_channel(AIChannel("assistant", provider=QuotaModel()))
    await kit.create_room(room_id="support")
    await kit.attach_channel("support", "sms")
    await kit.attach_channel("support", "assistant", category=ChannelCategory.INTELLIGENCE)
    result = await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="+15550100", content=TextContent(body="Hi"))
    )
    await kit.close()
    return result.error, reported


def show(error: BaseException | None, reported: list[str]) -> None:
    logger.info("  type: %s (reported as %s)", type(error).__name__, ", ".join(reported))
    logger.info("  a ProviderError: %s", isinstance(error, ProviderError))
    logger.info("  a RemoteTurnError: %s", isinstance(error, RemoteTurnError))
    logger.info("  code: %s, retry after: %s s", error.code, error.retry_after)  # type: ignore[union-attr]


async def main() -> None:
    logger.info("1. A kit that declares nothing:")
    show(*await ask(RoomKit()))

    logger.info("2. A kit that declares QuotaExceededError:")
    error, reported = await ask(RoomKit(error_classes=(QuotaExceededError,)))
    show(error, reported)
    logger.info("  the application's class itself: %s", type(error) is QuotaExceededError)


if __name__ == "__main__":
    asyncio.run(main())
