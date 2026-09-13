"""Proactive realtime requests identify a session and report actual injection."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from roomkit import Access, HookResult, HookTrigger, Immediate, Queued, RoomKit, WaitForIdle
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.delivery.base import DeliveryItem
from roomkit.delivery.worker import execute_delivery
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport


@asynccontextmanager
async def voice_room(count: int = 2):
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel("voice", provider=provider, transport=MockRealtimeTransport())
    async with RoomKit() as kit:
        kit.register_channel(channel)
        await kit.create_room(room_id="r")
        await kit.attach_channel("r", "voice")
        sessions = [
            await channel.start_session("r", f"person-{i}", object()) for i in range(count)
        ]
        yield kit, channel, provider, sessions


@pytest.mark.parametrize("strategy", [Immediate(), WaitForIdle(buffer=0), Queued(buffer=0)])
async def test_exact_session_does_not_inject_other_sessions(strategy) -> None:
    async with voice_room() as (kit, _, provider, sessions):
        result = await kit.deliver(
            "r",
            "external result",
            channel_id="voice",
            session_id=sessions[0].id,
            strategy=strategy,
            idempotency_key="external:1",
        )
        assert result.status == "sent"
        assert result.session_ids == [sessions[0].id]
        assert provider.injected_texts == [(sessions[0].id, "external result", "user")]
        assert result.reason == "voice_not_deduplicated"
        assert not result.turn_complete


async def test_explicit_channel_requires_unambiguous_session() -> None:
    async with voice_room() as (kit, _, provider, _):
        result = await kit.deliver("r", "external result", channel_id="voice")
        assert result.status == "unavailable"
        assert result.reason == "ambiguous_voice_session"
        assert provider.injected_texts == []


async def test_unaddressed_room_delivery_preserves_fanout() -> None:
    async with voice_room() as (kit, _, provider, sessions):
        result = await kit.deliver("r", "announcement")
        assert result.status == "sent"
        assert result.session_ids == [s.id for s in sessions]
        assert len(provider.injected_texts) == 2


@pytest.mark.parametrize("strategy", [WaitForIdle(buffer=0), Queued(buffer=0)])
async def test_session_replacement_while_waiting_is_not_success(strategy) -> None:
    async with voice_room(1) as (kit, channel, provider, sessions):

        async def replace_session(*args) -> None:
            await channel.end_session(sessions[0])
            await channel.start_session("r", "replacement", object())

        with patch("roomkit.core.delivery._wait_for_voice_idle", side_effect=replace_session):
            result = await kit.deliver("r", "result", channel_id="voice", strategy=strategy)
        assert result.status == "unavailable"
        assert result.reason == "voice_session_replaced"
        assert provider.injected_texts == []


async def test_missing_session_in_direct_and_worker_paths() -> None:
    async with voice_room(0) as (kit, _, provider, _):
        direct = await kit.deliver("r", "result", channel_id="voice", session_id="gone")
        worker = await execute_delivery(
            kit,
            DeliveryItem(room_id="r", content="result", channel_id="voice", session_id="gone"),
        )
        assert direct.status == worker.status == "unavailable"
        assert direct.reason == worker.reason == "voice_session_unavailable"
        assert provider.injected_texts == []


async def test_injection_error_can_retry_but_does_not_claim_deduplication() -> None:
    async with voice_room(1) as (kit, _, provider, sessions):
        item = DeliveryItem(
            room_id="r",
            content="result",
            channel_id="voice",
            session_id=sessions[0].id,
            idempotency_key="external:1",
        )
        with patch.object(
            provider, "inject_text", new=AsyncMock(side_effect=ConnectionError("offline"))
        ):
            failed = await execute_delivery(kit, item)
        assert failed.status == "failed"
        assert failed.reason == "voice_injection_failed"
        assert failed.error.message == "offline"
        assert failed.error.retryable
        retry = await execute_delivery(kit, item)
        assert retry.status == "sent"
        assert retry.reason == "voice_not_deduplicated"
        assert provider.injected_texts == [(sessions[0].id, "result", "user")]


@pytest.mark.parametrize("address", [[], ["missing-agent"]])
async def test_intelligence_address_never_becomes_direct_voice_injection(address) -> None:
    async with voice_room(1) as (kit, _, provider, _):
        first = await kit.deliver("r", "event", addressed_to=address, idempotency_key="k")
        replay = await kit.deliver("r", "event", addressed_to=address, idempotency_key="k")
        assert first.status == ("unavailable" if address else "sent")
        assert replay.duplicate and first.event_id == replay.event_id
        assert provider.injected_texts == []
        event = await kit.store.get_event_by_idempotency_key("r", "k")
        assert event.addressed_to == address


async def test_voice_refusal_permissions_and_muting() -> None:
    async with voice_room(1) as (kit, _, provider, _):
        await kit.set_access("r", "voice", Access.NONE)
        denied = await kit.deliver("r", "denied", channel_id="voice")
        assert denied.status == "blocked"
        assert provider.injected_texts == []
        await kit.set_access("r", "voice", Access.READ_WRITE)
        await kit.mute("r", "voice")
        with patch.object(provider, "inject_text", wraps=provider.inject_text) as inject:
            silent = await kit.deliver("r", "context", channel_id="voice")
        assert silent.status == "sent"
        assert inject.call_args.kwargs["silent"] is True

        @kit.hook(HookTrigger.BEFORE_DELIVER)
        async def refuse(event, context):
            return HookResult.block("policy")

        refused = await kit.deliver("r", "refused", channel_id="voice")
        assert refused.status == "blocked"
        assert len(provider.injected_texts) == 1
