"""A Gemini receive loop runs in a context of its own (RMK-280 follow-up).

``reconfigure`` restarts the loop from inside whatever called it, a tool
handler for a handoff; the new connection's events must not carry that call's
context (its voice session, its AI loop, the call it serves).
"""

from __future__ import annotations

import contextvars
from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai", reason="google-genai not installed")

from roomkit.providers.gemini.realtime import GeminiLiveProvider  # noqa: E402

_CALLER = contextvars.ContextVar("caller", default=None)


async def test_the_loop_does_not_inherit_its_starters_context() -> None:
    provider = GeminiLiveProvider(api_key="test-key")
    seen: list[str | None] = []

    async def receive_loop(session: object) -> None:
        seen.append(_CALLER.get())

    provider._receive_loop = receive_loop  # type: ignore[method-assign]
    state = SimpleNamespace(session=SimpleNamespace(id="s1"), receive_task=None)
    _CALLER.set("the handler's call")

    provider._start_receive_loop(state)  # type: ignore[arg-type]
    await state.receive_task

    assert seen == [None]
