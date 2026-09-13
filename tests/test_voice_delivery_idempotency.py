"""Keyed voice delivery executes one provider attempt per selected session."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from roomkit import Immediate, Queued, VoiceInjectionResult, WaitForIdle
from roomkit.store.base import ConversationStore
from tests.test_proactive_delivery_voice import voice_room


@pytest.mark.parametrize("strategy", [Immediate(), WaitForIdle(buffer=0), Queued(buffer=0)])
async def test_concurrent_keyed_calls_share_one_injection(strategy) -> None:
    async with voice_room() as (kit, _, provider, sessions):
        send = provider.inject_text

        async def slow_send(*args, **kwargs):
            await asyncio.sleep(0.01)
            return await send(*args, **kwargs)

        with patch.object(provider, "inject_text", side_effect=slow_send):
            results = await asyncio.gather(
                *[
                    kit.deliver(
                        "r",
                        "done",
                        channel_id="voice",
                        session_id=sessions[0].id,
                        idempotency_key="task:42",
                        strategy=strategy,
                    )
                    for _ in range(12)
                ]
            )
        assert {result.status for result in results} == {"sent"}
        assert sum(result.duplicate for result in results) == 11
        assert provider.injected_texts == [(sessions[0].id, "done", "user")]
        assert all(result.session_ids == [sessions[0].id] for result in results)
        assert all(not result.turn_complete for result in results)


async def test_known_result_survives_session_end_and_content_conflicts_are_refused() -> None:
    async with voice_room(1) as (kit, channel, provider, sessions):
        args = dict(channel_id="voice", session_id=sessions[0].id, idempotency_key="key")
        sent = await kit.deliver("r", "original", **args)
        await channel.end_session(sessions[0])
        replay = await kit.deliver("r", "original", **args)
        conflict = await kit.deliver("r", "changed", **args)
        assert sent.status == replay.status == "sent"
        assert replay.duplicate
        assert conflict.status == "blocked"
        assert conflict.reason == "voice_idempotency_conflict"
        assert provider.injected_texts == [(sessions[0].id, "original", "user")]


async def test_keys_and_sessions_are_independent_and_unkeyed_calls_are_not_deduplicated() -> None:
    async with voice_room() as (kit, _, provider, sessions):
        for session in sessions:
            for key in ("first", "second", None, None):
                result = await kit.deliver(
                    "r", "done", channel_id="voice", session_id=session.id, idempotency_key=key
                )
                assert result.status == "sent" and not result.duplicate
        assert len(provider.injected_texts) == 8


async def test_only_guaranteed_non_submission_can_be_retried() -> None:
    async with voice_room(1) as (kit, _, provider, sessions):
        args = dict(channel_id="voice", session_id=sessions[0].id, idempotency_key="key")
        refused = VoiceInjectionResult(status="not_sent", reason="not_connected", retryable=True)
        with patch.object(provider, "inject_text", new=AsyncMock(return_value=refused)):
            first = await kit.deliver("r", "done", **args)
        assert first.status == "failed" and first.error.retryable
        second = await kit.deliver("r", "done", **args)
        third = await kit.deliver("r", "done", **args)
        assert second.status == third.status == "sent"
        assert not second.duplicate and third.duplicate
        assert provider.injected_texts == [(sessions[0].id, "done", "user")]


@pytest.mark.parametrize("failure", ["exception", "queued", "unreported"])
async def test_uncertain_acceptance_never_reinjects(failure) -> None:
    async with voice_room(1) as (kit, _, provider, sessions):
        send = provider.inject_text

        async def uncertain(*args, **kwargs):
            await send(*args, **kwargs)
            if failure == "exception":
                raise ConnectionError("confirmation lost")
            if failure == "queued":
                return VoiceInjectionResult(status="unknown", reason="voice_provider_queued")
            return None

        args = dict(channel_id="voice", session_id=sessions[0].id, idempotency_key="key")
        with patch.object(provider, "inject_text", side_effect=uncertain):
            first = await kit.deliver("r", "done", **args)
        replay = await kit.deliver("r", "done", **args)
        assert first.status == replay.status == "unknown"
        assert not first.error.retryable and replay.duplicate
        assert len(provider.injected_texts) == 1


async def test_partial_fanout_retries_only_the_session_that_did_not_send() -> None:
    async with voice_room() as (kit, _, provider, sessions):
        send = provider.inject_text

        async def partial(session, *args, **kwargs):
            if session is sessions[1]:
                return VoiceInjectionResult(
                    status="not_sent", reason="disconnected", retryable=True
                )
            return await send(session, *args, **kwargs)

        with patch.object(provider, "inject_text", side_effect=partial):
            first = await kit.deliver("r", "done", idempotency_key="key")
        assert first.status == "failed"
        assert first.session_ids == [sessions[0].id]
        assert first.session_outcomes[sessions[0].id].status == "sent"
        replay = await kit.deliver("r", "done", idempotency_key="key")
        assert replay.status == "sent" and not replay.duplicate
        assert replay.session_outcomes[sessions[0].id].duplicate
        assert not replay.session_outcomes[sessions[1].id].duplicate
        assert provider.injected_texts == [(s.id, "done", "user") for s in sessions]


async def test_reentrant_same_key_does_not_deadlock_or_start_another_attempt() -> None:
    async with voice_room(1) as (kit, _, provider, sessions):
        send = provider.inject_text
        args = dict(channel_id="voice", session_id=sessions[0].id, idempotency_key="key")
        nested = []

        async def reenter(*values, **kwargs):
            nested.append(await kit.deliver("r", "done", **args))
            return await send(*values, **kwargs)

        with patch.object(provider, "inject_text", side_effect=reenter):
            result = await asyncio.wait_for(kit.deliver("r", "done", **args), 1)
        assert result.status == "sent"
        assert nested[0].status == "unknown" and nested[0].duplicate
        assert len(provider.injected_texts) == 1


async def test_unsupported_store_refuses_keyed_voice_before_sending() -> None:
    async with voice_room(1) as (kit, _, provider, sessions):

        async def unsupported(*args):
            return await ConversationStore.get_voice_delivery(kit.store, *args)

        with patch.object(kit.store, "get_voice_delivery", side_effect=unsupported):
            blocked = await kit.deliver(
                "r", "done", channel_id="voice", session_id=sessions[0].id, idempotency_key="key"
            )
            assert blocked.status == "blocked"
            assert blocked.reason == "voice_idempotency_unsupported"
            assert provider.injected_texts == []
            plain = await kit.deliver("r", "done", channel_id="voice", session_id=sessions[0].id)
            assert plain.status == "sent"
