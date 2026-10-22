"""A redelivered message observes what the first delivery did (RFC §13.4).

"If seen → return the original result without reprocessing." The check honoured
the second half and answered the first with ``blocked=True, reason="duplicate"``
and no event — so a sender retrying because it never saw the first response
learned only that its message had been refused, which is the one thing that had
not happened.
"""

from __future__ import annotations

import pytest

from roomkit import RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import SQLiteStore
from tests.test_framework import SimpleChannel


async def _kit(store=None) -> RoomKit:  # noqa: ANN001
    kit = RoomKit(store=store) if store is not None else RoomKit()
    kit.register_channel(SimpleChannel("ch-1", ChannelType.SMS))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ch-1")
    return kit


def _message(
    text: str = "hello", *, key: str | None = "k-1", channel_id: str = "ch-1"
) -> InboundMessage:
    return InboundMessage(
        channel_id=channel_id,
        sender_id="user-1",
        content=TextContent(body=text),
        idempotency_key=key,
    )


class TestTheSecondDeliveryAnswersWithTheFirst:
    async def test_the_duplicate_returns_the_committed_event(self) -> None:
        kit = await _kit()

        first = await kit.process_inbound(_message())
        second = await kit.process_inbound(_message())

        assert first.event is not None
        assert second.event is not None
        assert second.event.id == first.event.id
        assert second.event.index == first.event.index

    async def test_the_duplicate_does_not_read_as_a_refusal(self) -> None:
        kit = await _kit()

        await kit.process_inbound(_message())
        second = await kit.process_inbound(_message())

        assert second.blocked is False
        assert second.reason is None

    async def test_nothing_is_reprocessed(self) -> None:
        """The MUST that was already honoured stays honoured."""
        kit = await _kit()

        await kit.process_inbound(_message())
        before = len(await kit.get_timeline("r1"))
        await kit.process_inbound(_message("a different body"))

        assert len(await kit.get_timeline("r1")) == before

    async def test_a_distinct_key_is_processed_normally(self) -> None:
        kit = await _kit()

        first = await kit.process_inbound(_message(key="k-1"))
        second = await kit.process_inbound(_message(key="k-2"))

        assert first.event is not None
        assert second.event is not None
        assert second.event.id != first.event.id

    async def test_no_key_means_no_idempotency(self) -> None:
        kit = await _kit()

        first = await kit.process_inbound(_message(key=None))
        second = await kit.process_inbound(_message(key=None))

        assert first.event is not None
        assert second.event is not None
        assert second.event.id != first.event.id


class TestAKeyIsUniquePerSourceChannel:
    """Two channels of one room may carry one key for two messages (RFC §13.4)."""

    @pytest.mark.parametrize("backend", ["memory", "sqlite"])
    async def test_two_channels_sending_one_key_are_both_processed(
        self,
        backend: str,
        tmp_path,  # noqa: ANN001
    ) -> None:
        kit = await _kit(store=SQLiteStore(tmp_path / "rooms.db") if backend == "sqlite" else None)
        kit.register_channel(SimpleChannel("ch-2", ChannelType.SMS))
        await kit.attach_channel("r1", "ch-2")

        first = await kit.process_inbound(_message("from one"))
        second = await kit.process_inbound(_message("from two", channel_id="ch-2"))

        assert first.event is not None and second.event is not None
        assert (second.duplicate, second.event.id != first.event.id) == (False, True)

    async def test_the_same_channel_sending_the_key_again_is_a_replay(self) -> None:
        kit = await _kit()
        kit.register_channel(SimpleChannel("ch-2", ChannelType.SMS))
        await kit.attach_channel("r1", "ch-2")
        await kit.process_inbound(_message("from one"))
        await kit.process_inbound(_message("from two", channel_id="ch-2"))

        again = await kit.process_inbound(_message("from two", channel_id="ch-2"))

        assert again.duplicate is True
        assert isinstance(again.event.content, TextContent)
        assert again.event.content.body == "from two"


class TestABlockedMessageRedelivered:
    async def test_returns_the_blocked_record_with_its_reason(self) -> None:
        kit = await _kit()

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="no_spam")
        async def no_spam(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("looks like spam")

        first = await kit.process_inbound(_message())
        second = await kit.process_inbound(_message())

        assert first.event is not None and second.event is not None
        assert (second.blocked, second.duplicate) == (True, True)
        assert (second.event.id, second.reason) == (first.event.id, "no_spam")


class TestAStoreThatCannotResolveTheKey:
    async def test_falls_back_to_the_blocked_result(self) -> None:
        """``get_event_by_idempotency_key`` defaults to ``None`` on the ABC, so a
        store written before it existed keeps the answer it always gave."""

        class _OlderStore(InMemoryStore):
            async def get_event_by_idempotency_key(  # noqa: ANN202
                self, room_id: str, key: str, *, channel_id: str
            ):
                return None

        kit = await _kit(store=_OlderStore())

        await kit.process_inbound(_message())
        second = await kit.process_inbound(_message())

        assert second.blocked is True
        assert second.reason == "duplicate"
        assert second.event is None


class TestSQLite:
    async def test_the_duplicate_returns_the_committed_event(self, tmp_path) -> None:  # noqa: ANN001
        kit = await _kit(store=SQLiteStore(tmp_path / "rooms.db"))

        first = await kit.process_inbound(_message())
        second = await kit.process_inbound(_message())

        assert first.event is not None
        assert second.event is not None
        assert second.event.id == first.event.id


class TestTheStoreLookup:
    async def test_a_recorded_key_resolves_to_its_event(self) -> None:
        kit = await _kit()
        result = await kit.process_inbound(_message())
        assert result.event is not None

        found = await kit.store.get_event_by_idempotency_key("r1", "k-1", channel_id="ch-1")

        assert found is not None
        assert found.id == result.event.id

    async def test_an_unknown_key_resolves_to_nothing(self) -> None:
        kit = await _kit()
        await kit.process_inbound(_message())

        assert (
            await kit.store.get_event_by_idempotency_key("r1", "never-sent", channel_id="ch-1")
            is None
        )

    async def test_a_key_is_scoped_to_its_room(self) -> None:
        kit = await _kit()
        await kit.create_room(room_id="r2")
        await kit.process_inbound(_message())

        assert await kit.store.get_event_by_idempotency_key("r2", "k-1", channel_id="ch-1") is None
