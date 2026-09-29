"""What a provider hands the tool loop for a call, the same on every provider.

Each provider reads its own wire, but the loop must receive one thing for one
call (RFC §6.4): a mapping of arguments and never an error, an id no other
call of the response carries, and a mark on a call the output cap cut before
its arguments were complete. The rules live here once, so a provider cannot
drift from the others by reading its wire its own way.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

# OpenAI-compatible servers and Ollama's ``done_reason`` say ``length``,
# Anthropic's ``stop_reason`` says ``max_tokens``, Gemini's candidate says
# ``MAX_TOKENS``. A rule that knew only one spelling would cover only the
# providers using it.
_TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens"})


def is_truncation(finish_reason: str | None) -> bool:
    """Whether a response ended by exhausting its output budget.

    Compared case-insensitively so Gemini's ``MAX_TOKENS`` and Anthropic's
    ``max_tokens`` are one entry rather than two.
    """
    return finish_reason is not None and finish_reason.lower() in _TRUNCATION_FINISH_REASONS


def tool_arguments(raw: Any) -> dict[str, Any]:
    """A call's arguments as a mapping, never an error.

    A mapping passes through. Empty arguments are ``{}``. Text that parses to a
    JSON object is that object; anything else (invalid JSON, ``null``, an
    array, a fragment the output cap cut) is kept whole under ``raw``.
    """
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if not isinstance(raw, str):
        return {"raw": raw}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"raw": raw}


def arguments_cut(raw: Any) -> bool:
    """Whether argument text stops before its JSON ends.

    What the output cap leaves of a call it cut; complete JSON that is not an
    object (``null``, an array) is whole, however unusable.
    """
    if not isinstance(raw, str) or not raw.strip():
        return False
    try:
        json.loads(raw)
    except ValueError:
        return True
    return False


def minted_call_id(name: str) -> str:
    """An id for a call its server gave none, unique across turns."""
    return f"call_{name}_{uuid4().hex[:12]}"


class CallIds:
    """Hands each call of one response its id.

    The server's id when it gave one no earlier call of the response took,
    a minted one otherwise: results, eviction and cancellation all find a
    call by its id, so two calls must never share one.
    """

    def __init__(self) -> None:
        self._taken: set[str] = set()

    def __call__(self, server_id: str | None, name: str) -> str:
        call_id = server_id if server_id and server_id not in self._taken else None
        call_id = call_id or minted_call_id(name)
        self._taken.add(call_id)
        return call_id
