"""The realtime tool calls in flight on a channel, one record per call (RFC §12.4).

Every door of a speech-to-speech channel (the provider's function call, a call
recovered from speech, a reasoning backend's call) opens a record here when a
call arrives and closes it when the call ends. The record is where the call's
one delivery and one report are claimed, so a reconfiguration that fails once
the result went out, a cancellation that lands after it, or the session's end
cannot add a second outcome to the same call.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.providers.ai.tool_calls import tool_arguments, unreadable_call_error

if TYPE_CHECKING:
    from roomkit.tools._outcome import ToolOutcome
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.channels.realtime_tools")


@dataclass(eq=False)
class RealtimeToolCall:
    """One realtime tool call: what was called, the task serving it, and
    whether its result went out and its outcome was reported."""

    session: VoiceSession
    call_id: str
    name: str
    arguments: dict[str, Any]
    room_id: str | None = None
    """The room the session served when the call ran."""
    unreadable: str | None = None
    """What the model reads when the call cannot be read (its arguments, or a
    ``call_tool`` transport's, are not an object): refused before the gate."""
    mutes: bool = False
    """The call holds the session's input muted while it runs."""
    structured_content: dict[str, Any] | None = None
    """The structured copy its handler left on the tool call context (MCP
    ``structuredContent``), carried to ON_TOOL_CALL (RFC §9.3)."""
    task: asyncio.Task[Any] | None = field(default=None, repr=False)
    delivered: bool = False
    reported: bool = False
    owed: ToolOutcome | None = None
    """The outcome the model reads, kept from its delivery when the call's
    report comes after it: a report an ending cuts still owes it (RFC §9.3)."""

    @classmethod
    def from_provider(
        cls,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any] | str,
        **fields: Any,
    ) -> RealtimeToolCall:
        """The call a provider handed ``on_tool_call``: arguments that came as
        the model's text did not read as an object, and the call is
        unreadable, kept under ``raw`` for its reports (RFC §6.4, §12.4)."""
        if isinstance(arguments, dict):
            return cls(session, call_id, name, arguments, **fields)
        logger.warning(
            "Provider sent unreadable arguments for tool call %s (%s): it does not run",
            name,
            call_id,
        )
        refusal = json.dumps(unreadable_call_error(name))
        return cls(session, call_id, name, tool_arguments(arguments), unreadable=refusal, **fields)

    def claim_report(self) -> bool:
        """Claim the call's one report: False when it was already made."""
        if self.reported:
            return False
        self.reported = True
        return True


class ToolCallBook:
    """The realtime tool calls in flight, per session."""

    def __init__(self) -> None:
        self._calls: dict[str, dict[str, RealtimeToolCall]] = {}

    def open(self, call: RealtimeToolCall) -> bool:
        """Record *call*: False when a call with its id is already in flight,
        which then keeps the id (RFC §12.4)."""
        calls = self._calls.setdefault(call.session.id, {})
        if call.call_id in calls:
            return False
        calls[call.call_id] = call
        return True

    def close(self, call: RealtimeToolCall) -> None:
        """Forget *call*, if it is still the one its id names."""
        calls = self._calls.get(call.session.id)
        if calls is not None and calls.get(call.call_id) is call:
            del calls[call.call_id]
            if not calls:
                del self._calls[call.session.id]

    def get(self, session_id: str, call_id: str) -> RealtimeToolCall | None:
        return (self._calls.get(session_id) or {}).get(call_id)

    def abandonable(self, session_id: str, call_id: str) -> RealtimeToolCall | None:
        """The call a provider cancellation for *call_id* interrupts: in
        flight, its task running, its result not out, its outcome not
        reported. ``None`` when there is nothing left to interrupt."""
        call = self.get(session_id, call_id)
        if call is None or call.delivered or call.reported:
            return None
        if call.task is None or call.task.done():
            return None
        return call

    def busy(self, session_id: str) -> bool:
        """Whether a call is in flight on the session."""
        return bool(self._calls.get(session_id))

    def muting(self, session_id: str) -> bool:
        """Whether a call holding the input muted is in flight on the session."""
        return any(call.mutes for call in (self._calls.get(session_id) or {}).values())

    def take(self, session_id: str) -> list[RealtimeToolCall]:
        """Every call in flight on the session, off the books: the session ends."""
        return list(self._calls.pop(session_id, {}).values())
