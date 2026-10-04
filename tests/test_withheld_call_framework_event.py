"""The ``tool_call`` framework event of a call ON_TOOL_CALL withheld says it
failed, on every door (RMK-480, RFC §9.3).

A SYNC hook's BLOCK, or a fail-closed hook that raises, withholds the result:
the observers and the model read the failure, and so does the framework
event, which is the call's only report when the hooks' context will not
build.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from tests.tool_doors import DOORS, Hooks, run_door

EVERY_DOOR = pytest.mark.parametrize("door", DOORS)


async def _served(name: str, arguments: dict[str, Any]) -> str:
    return "secret result"


async def _block(event: Any, ctx: Any) -> HookResult:
    return HookResult.block("withheld")


def _fail_closed(*, context_fails: bool) -> Any:
    """A fail-closed ON_TOOL_CALL hook that raises, or whose context will not
    build for the call."""

    def setup(kit: RoomKit) -> None:
        @kit.hook(
            HookTrigger.ON_TOOL_CALL,
            execution=HookExecution.SYNC,
            name="redact",
            fail_closed=True,
        )
        async def redact(event: Any, ctx: Any) -> HookResult:
            if not context_fails:
                raise RuntimeError("redactor down")
            return HookResult.allow()

        if context_fails:
            real_chain = kit._run_tool_call_chain
            real_build = kit._build_context
            armed = False

            async def build(room_id: str, **kwargs: Any) -> Any:
                if armed:
                    raise RuntimeError("store down")
                return await real_build(room_id, **kwargs)

            async def chain(event: Any, room_id: str, **kwargs: Any) -> Any:
                nonlocal armed
                armed = True
                try:
                    return await real_chain(event, room_id, **kwargs)
                finally:
                    armed = False

            kit._build_context = build  # type: ignore[method-assign]
            kit._run_tool_call_chain = chain  # type: ignore[method-assign]

    return setup


@EVERY_DOOR
async def test_a_blocked_call_s_framework_event_says_it_failed(door: str) -> None:
    seen = await run_door(door, _served, hooks=Hooks(judge=_block))

    assert [event.get("is_error") for event in seen.framework] == [True]
    assert [event.is_error for event in seen.reports] == [True]


@EVERY_DOOR
@pytest.mark.parametrize("context_fails", [False, True], ids=["hook-raises", "context-fails"])
async def test_a_fail_closed_withheld_call_s_framework_event_says_it_failed(
    door: str, context_fails: bool
) -> None:
    hooks = Hooks(sync_hook=False, setup=_fail_closed(context_fails=context_fails))
    seen = await run_door(door, _served, hooks=hooks)

    assert [event.get("is_error") for event in seen.framework] == [True]
