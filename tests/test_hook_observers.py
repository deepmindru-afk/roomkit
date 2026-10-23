"""ASYNC observers of a SYNC-run trigger run without holding up the operation (RFC §9, RMK-903)."""

from __future__ import annotations

import asyncio
import logging

import pytest

from roomkit import RoomKit
from roomkit.core.hooks import HookEngine, HookRegistration
from roomkit.models.context import RoomContext
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.models.hook import HookResult
from roomkit.models.room import Room


def _ctx() -> RoomContext:
    return RoomContext(room=Room(id="r1"))


def _register(engine: HookEngine, execution: HookExecution, fn, name: str) -> None:
    engine.register(
        HookRegistration(trigger=HookTrigger.BEFORE_TTS, execution=execution, fn=fn, name=name)
    )


class TestTheOperationGoesOn:
    async def test_a_sync_run_trigger_returns_before_a_slow_observer_ends(self) -> None:
        engine = HookEngine()
        seen: list[str] = []
        release = asyncio.Event()

        async def slow_observer(text: str, ctx: RoomContext) -> None:
            await release.wait()
            seen.append(text)

        _register(engine, HookExecution.ASYNC, slow_observer, "slow")
        result = await asyncio.wait_for(
            engine.run_sync_hooks(
                "r1", HookTrigger.BEFORE_TTS, "hello", _ctx(), skip_event_filter=True
            ),
            1,
        )
        assert result.allowed and seen == []  # returned while the observer still waits

        release.set()
        await engine.drain_observers()
        assert seen == ["hello"]  # it ran all the same

    async def test_run_observers_returns_at_once_too(self) -> None:
        engine = HookEngine()
        done, release = asyncio.Event(), asyncio.Event()

        async def slow_observer(text: str, ctx: RoomContext) -> None:
            await release.wait()
            done.set()

        _register(engine, HookExecution.ASYNC, slow_observer, "slow")
        await asyncio.wait_for(
            engine.run_observers(
                "r1", HookTrigger.BEFORE_TTS, "hello", _ctx(), skip_event_filter=True
            ),
            1,
        )
        assert not done.is_set()
        release.set()
        await engine.drain_observers()
        assert done.is_set()

    async def test_the_observer_sees_what_the_sync_chain_left(self) -> None:
        engine = HookEngine()
        seen: list[str] = []

        async def rewrite(text: str, ctx: RoomContext) -> HookResult:
            return HookResult.modify(text.upper())

        async def observer(text: str, ctx: RoomContext) -> None:
            seen.append(text)

        _register(engine, HookExecution.SYNC, rewrite, "rewrite")
        _register(engine, HookExecution.ASYNC, observer, "observer")
        await engine.run_sync_hooks(
            "r1", HookTrigger.BEFORE_TTS, "hello", _ctx(), skip_event_filter=True
        )
        await engine.drain_observers()
        assert seen == ["HELLO"]

    async def test_an_observer_that_fails_is_logged_and_nothing_else(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine = HookEngine()

        async def failing(text: str, ctx: RoomContext) -> None:
            raise RuntimeError("observer broke")

        _register(engine, HookExecution.ASYNC, failing, "failing")
        with caplog.at_level(logging.ERROR, logger="roomkit.hooks"):
            result = await engine.run_sync_hooks(
                "r1", HookTrigger.BEFORE_TTS, "hello", _ctx(), skip_event_filter=True
            )
            await engine.drain_observers()
        assert result.allowed
        assert any("Async hook failing failed" in r.getMessage() for r in caplog.records)


class TestClose:
    async def test_close_waits_for_the_observers_still_running(self) -> None:
        kit = RoomKit()
        room = await kit.create_room()
        finished, release = asyncio.Event(), asyncio.Event()

        @kit.hook(HookTrigger.BEFORE_TTS, execution=HookExecution.ASYNC)
        async def slow_observer(text: str, ctx: RoomContext) -> None:
            await release.wait()
            finished.set()

        await kit.hook_engine.run_sync_hooks(
            room.id, HookTrigger.BEFORE_TTS, "hello", _ctx(), skip_event_filter=True
        )
        assert not finished.is_set()
        closing = asyncio.ensure_future(kit.close())
        await asyncio.wait({closing}, timeout=0.05)
        assert not closing.done()  # it waits for the observer still running
        release.set()
        await asyncio.wait_for(closing, 5)
        assert finished.is_set()
        assert not kit.hook_engine._observers
