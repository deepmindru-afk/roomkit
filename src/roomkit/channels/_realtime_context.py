"""Context variables for RealtimeVoiceChannel tool calls."""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.base import VoiceSession

_current_voice_session: contextvars.ContextVar[VoiceSession | None] = contextvars.ContextVar(
    "_current_voice_session",
    default=None,
)


def get_current_voice_session() -> VoiceSession | None:
    """Get the voice session for the current tool call.

    Available inside tool handlers called by RealtimeVoiceChannel.
    Returns None outside of a tool call context.
    """
    return _current_voice_session.get()


class _ServedCall:
    """The provider call a tool-call task is serving.

    A reconnect its own handler causes (a handoff reconfiguring its session)
    orphans it like every other call, but the model did not abandon it: the
    handler runs on, and ``orphaned`` records that its result has nowhere to
    go, since the new socket never issued the id (RFC §9.3). A task the
    handler starts inherits the record, and one can outlive the call (a
    provider's new receive loop does); ``finished`` keeps it from naming a
    call that has ended.
    """

    __slots__ = ("call_id", "finished", "orphaned", "session_id")

    def __init__(self, session_id: str, call_id: str) -> None:
        self.session_id = session_id
        self.call_id = call_id
        self.orphaned = False
        self.finished = False


_served_call: contextvars.ContextVar[_ServedCall | None] = contextvars.ContextVar(
    "_served_call",
    default=None,
)


@contextlib.contextmanager
def serving_call(session_id: str, call_id: str) -> Iterator[None]:
    """Run a provider call's handling as that call's own context."""
    served = _ServedCall(session_id, call_id)
    token = _served_call.set(served)
    try:
        yield
    finally:
        served.finished = True
        _served_call.reset(token)


def _this_task_serves(session_id: str, call_id: str) -> _ServedCall | None:
    served = _served_call.get()
    if served is None or served.finished:
        return None
    if (served.session_id, served.call_id) != (session_id, call_id):
        return None
    return served


def spare_own_orphaned_call(session_id: str, call_id: str) -> bool:
    """Whether the orphaned call is the one this task's handler is serving.

    Called where a provider reports orphaned calls. When the report runs
    inside the call's own handler, or a task it started, that handler caused
    the reconnect: the call is marked so that its result is not sent, and it
    is not to be interrupted.
    """
    served = _this_task_serves(session_id, call_id)
    if served is None:
        return False
    served.orphaned = True
    return True


def own_call_orphaned(session_id: str, call_id: str) -> bool:
    """Whether this task's call lost its id to a reconnect its handler caused."""
    served = _this_task_serves(session_id, call_id)
    return served is not None and served.orphaned
