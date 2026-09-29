"""Bookkeeping of a model response's function calls, shared by the OpenAI wires.

Both the classic Realtime wire and the GPT-Live hosted delegation ask the
model to go on with ``response.create`` once tool results are in, and both
must ask once per response, after it has ended and every call it emitted has
its result (RFC §12.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PendingResponse:
    """A model response whose function calls are being collected.

    The classic Realtime service rejects ``response.create`` while a response
    is active; the hosted delegation's service rejects it while any call of
    the run is unanswered. The model is asked to go on once the response is
    ``settled``, and only if it asked for a call at all. Kept per session by
    the classic wire, per delegated run by the hosted backend.
    """

    call_ids: set[str] = field(default_factory=set)
    had_calls: bool = False
    finished: bool = False
    # A continuation this client asked for and the server has not begun yet
    requested: bool = False

    @property
    def settled(self) -> bool:
        """The response has ended and every call it emitted has its result."""
        return self.finished and not self.call_ids

    @property
    def ready_to_continue(self) -> bool:
        return self.settled and self.had_calls
