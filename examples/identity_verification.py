"""Identity verification: the person writing has proven who they are (RFC §11.7).

A number that resolves to Alice says the phone is Alice's, not that Alice is
typing. With a ``verification_policy``, RoomKit labels every message of an
identified sender with their verification (``source.verification``), holds an
unverified one from the agents, and releases it once they prove it. How a
person proves it stays yours: here a page behind a one-time link checks a PIN,
and a code replied in the conversation.

Shows:
- an ON_VERIFICATION_REQUIRED hook that sends a link unless one is out
  (``kit.sender_verification``), and lets a general question through
  (``HookResult.allow_unverified()``)
- the bank's page: a wrong PIN counts against the request
  (``record_failed_attempt``), the right one completes it
  (``complete_verification``); the held question is then answered
- a code replied in the conversation: consumed, never stored
- an advisor ending a verification, and the ON_VERIFICATION_ENDED hook

Runs with mock providers, no key needed:
    uv run python examples/identity_verification.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from shared import setup_logging

from roomkit import (
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    VerificationPolicy,
)
from roomkit.channels import SMSChannel, WebSocketChannel
from roomkit.channels.ai import AIChannel
from roomkit.identity.mock import MockIdentityResolver
from roomkit.models.identity import Identity
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider

logger = setup_logging("identity_verification")

ALICE = "+15550000001"
PINS = {"id-alice": "1111"}  # the bank's own check, standing for a real login


class Bank:
    """A kit whose hooks ask for verification, and the bank's page behind the link."""

    def __init__(self, method: str = "link") -> None:
        self.sms = MockSMSProvider()
        self.ai = MockAIProvider(responses=["We open at 9.", "Your transfer went through."])
        self.kit = RoomKit(
            identity_resolver=MockIdentityResolver(
                mapping={ALICE: Identity(id="id-alice", display_name="Alice")}
            ),
            verification_policy=VerificationPolicy(),
        )
        self.method = method
        self.links: dict[str, str] = {}  # request id -> secret, as the link carries them
        self._sending: set[asyncio.Task[Any]] = set()
        self.kit.register_channel(SMSChannel("sms", provider=self.sms))
        self.kit.register_channel(AIChannel("assistant", provider=self.ai))
        self.kit.register_channel(WebSocketChannel("bank"))  # the bank's own messages
        self._hooks()

    def _hooks(self) -> None:
        kit = self.kit

        @kit.hook(HookTrigger.ON_ROOM_CREATED)
        async def joins(event: Any, ctx: Any) -> None:
            await kit.attach_channel(ctx.room.id, "assistant")
            await kit.attach_channel(ctx.room.id, "bank")

        @kit.hook(HookTrigger.ON_VERIFICATION_REQUIRED)
        async def ask_to_verify(event: Any, ctx: Any) -> HookResult:
            if "open" in event.content.body:
                return HookResult.allow_unverified()  # a general question goes through
            state = await kit.sender_verification(event, ctx)
            if state is not None and state.request_id is None:
                request, secret = await kit.request_verification(
                    state.identity_id,
                    room_id=event.room_id,
                    channel_id=event.source.channel_id,
                    method=self.method,
                    window_start=event.created_at,
                )
                self.links[request.id] = secret
                text = (
                    f"Reply with {secret} to confirm it's you."
                    if self.method == "code"
                    else f"Confirm it's you: https://bank.example/verify/{request.id}/{secret}"
                )
                self.send_later(event.room_id, text)  # the room is locked while hooks run
            return HookResult.allow()  # held: the agent does not read it yet

        @kit.hook(HookTrigger.ON_VERIFICATION_ENDED, execution=HookExecution.ASYNC)
        async def ended(event: Any, ctx: Any) -> None:
            data = event.content.data
            logger.info("verification ended: %s (%s)", data["reason"], data["ended"])

    def send_later(self, room_id: str, text: str) -> None:
        # A message of the room: stored like any other, so the secret it carries
        # is readable by whoever reads the timeline. To keep it out, send it
        # through the provider directly.
        task = asyncio.create_task(
            self.kit.send_event(room_id, "bank", TextContent(body=text), visibility="sms")
        )
        self._sending.add(task)
        task.add_done_callback(self._sending.discard)

    async def page(self, request_id: str, pin: str) -> str:
        """The bank's page: its own check first, then RoomKit's."""
        request = await self.kit.store.get_verification_request(request_id)
        assert request is not None
        if pin != PINS[request.identity_id]:
            counted = await self.kit.record_failed_attempt(request_id)
            return f"wrong PIN ({counted.max_attempts - counted.attempts} attempt(s) left)"
        await self.kit.complete_verification(request_id, self.links[request_id])
        return "verified"

    async def text(self, body: str) -> None:
        await self.kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id=ALICE, content=TextContent(body=body))
        )
        await asyncio.sleep(0.2)  # let the bank's message and the agent's answer go out

    def phone(self) -> list[str]:
        sent = [m["event"] for m in self.sms.sent if m["to"] == ALICE]
        return [event.content.body for event in sent]  # type: ignore[union-attr]


async def by_link() -> None:
    logger.info("--- A link to the bank's page ---")
    bank = Bank()
    await bank.text("When do you open?")
    await bank.text("Did my transfer go through?")
    logger.info("the agent answered %d message(s) so far", len(bank.ai.calls))

    request_id = next(iter(bank.links))
    logger.info("page: %s", await bank.page(request_id, "0000"))
    logger.info("page: %s", await bank.page(request_id, "1111"))
    await asyncio.sleep(0.3)  # the held question is released to the agent
    for body in bank.phone():
        logger.info("alice's phone: %s", body)

    await bank.kit.revoke_verification("id-alice", reason="advisor")
    await asyncio.sleep(0.1)
    await bank.kit.close()


async def by_code() -> None:
    logger.info("--- A code replied in the conversation ---")
    bank = Bank(method="code")
    await bank.text("Did my transfer go through?")
    code = next(iter(bank.links.values()))
    await bank.text(code)
    await asyncio.sleep(0.3)
    state = await bank.kit.verification_of("id-alice")
    logger.info(
        "after the reply: %s; the agent answered %d time(s)", state.status, len(bank.ai.calls)
    )
    room = (await bank.kit.store.list_rooms())[0]
    replies = [
        getattr(e.content, "body", "")
        for e in await bank.kit.store.list_events(room.id, limit=50)
        if e.source.channel_id == "sms"
    ]
    logger.info("alice's reply is stored: %s", code in replies)
    await bank.kit.close()


async def main() -> None:
    await by_link()
    await by_code()


if __name__ == "__main__":
    asyncio.run(main())
