"""What a provider hands the tool loop for a call, the same on every provider.

Each provider reads its own wire, but the loop must receive one thing for one
call (RFC §6.4): a mapping of arguments and never an error, an id no other
call of the response carries, and a mark on a call whose arguments do not
read, which never runs. The rules live here once, so a provider cannot drift
from the others by reading its wire its own way.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

# Every provider reports "I hit the output cap" in its own vocabulary, and
# RoomKit forwards the raw value rather than inventing a normalized one.
# OpenAI-compatible servers and Ollama's ``done_reason`` say ``length``,
# Anthropic's ``stop_reason`` says ``max_tokens``, Gemini's candidate says
# ``MAX_TOKENS``. A rule that knew only one spelling would cover only the
# providers using it.
_TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens"})

# Endings that can stop a call mid-arguments: the output cap under its
# spellings, Mistral's context cap, and a content filter cutting the stream.
_CALL_CUTTING_FINISH_REASONS = _TRUNCATION_FINISH_REASONS | {"model_length", "content_filter"}


# Endings where the model tried to call a tool and the provider could not
# parse the call, so none reached the loop (Gemini's MALFORMED_FUNCTION_CALL).
_MALFORMED_CALL_FINISH_REASONS = frozenset({"malformed_function_call"})


MALFORMED_CALL_NUDGE = (
    "Your last tool call could not be parsed, so it did not run. Call the tool "
    "again with arguments that are valid JSON matching its parameters, or answer "
    "in plain text."
)
"""What a model is told when its provider could not parse its tool call, on
the text loops and on a speech-to-speech session alike (RFC §6.4, §12.4)."""


def is_malformed_call(finish_reason: str | None) -> bool:
    """Whether a response ended on a tool call its provider could not parse."""
    return finish_reason is not None and finish_reason.lower() in _MALFORMED_CALL_FINISH_REASONS


def is_truncation(finish_reason: str | None) -> bool:
    """Whether a response ended by exhausting its output budget.

    Compared case-insensitively so Gemini's ``MAX_TOKENS`` and Anthropic's
    ``max_tokens`` are one entry rather than two.
    """
    return finish_reason is not None and finish_reason.lower() in _TRUNCATION_FINISH_REASONS


def tool_arguments(raw: Any) -> dict[str, Any]:
    """A call's arguments as a mapping, never an error.

    A mapping passes through. No arguments (nothing, blank text, JSON
    ``null``) are ``{}``. Text that parses to a JSON object is that object;
    anything else (invalid JSON, an array, a fragment the output cap cut) is
    kept whole under ``raw``, and the call is ``partial``
    (:func:`unreadable_arguments`).
    """
    read = _read_arguments(raw)
    return {"raw": raw} if read is None else read


def unreadable_arguments(raw: Any) -> bool:
    """Whether a call's arguments do not read as an object, which makes the
    call ``partial``: it never runs, whatever the provider and whatever stop
    reason the response gave (RFC §6.4)."""
    return _read_arguments(raw) is None


def _read_arguments(raw: Any) -> dict[str, Any] | None:
    """A call's arguments as a mapping, or ``None`` when they do not read as one."""
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    if parsed is None:
        return {}
    return parsed if isinstance(parsed, dict) else None


def arguments_cut(raw: Any) -> bool:
    """Whether argument text is not valid JSON: what a cut leaves of a call
    still being written. Complete JSON that is not an object (an array) is
    whole, however unusable."""
    if not isinstance(raw, str) or not raw.strip():
        return False
    try:
        json.loads(raw)
    except (ValueError, RecursionError):
        return True
    return False


def call_cut(raw: Any, finish_reason: str | None) -> bool:
    """Whether the response was cut short over a call's arguments: they do not
    read, and the response ended on something that stops a call mid-arguments
    (the output cap, a content filter) or on nothing at all, a stream that
    stopped without a stop reason. The call is ``partial`` and ``cut``."""
    if finish_reason is not None and finish_reason.lower() not in _CALL_CUTTING_FINISH_REASONS:
        return False
    return unreadable_arguments(raw)


def call_garbled(raw: Any, finish_reason: str | None) -> bool:
    """Whether the model wrote a call's arguments unreadable: they do not
    read, and the response ended on its own, not cut short over them
    (:func:`call_cut`). The call is ``partial`` and ``garbled``."""
    return unreadable_arguments(raw) and not call_cut(raw, finish_reason)


def partial_call_error(name: str, *, garbled: bool) -> dict[str, Any]:
    """What the model reads for a ``partial`` call: nothing ran, and why."""
    return unreadable_call_error(name) if garbled else cut_call_error(name)


def cut_call_error(name: str) -> dict[str, Any]:
    """What the model reads for a call the response cut: nothing ran."""
    return {
        "error": "Tool call cut off",
        "tool": name,
        "hint": (
            "This call was cut off before its arguments were complete, so it did "
            "not run. Call it again, with shorter arguments if you can."
        ),
    }


def unreadable_call_error(name: str) -> dict[str, Any]:
    """What the model reads for a call written with unreadable arguments:
    nothing ran."""
    return {
        "error": "Tool call arguments unreadable",
        "tool": name,
        "hint": (
            "This call's arguments are not a JSON object, so it did not run. Call "
            "it again with arguments that are a valid JSON object matching the "
            "tool's parameters."
        ),
    }


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

    def __call__(self, server_id: Any, name: str) -> str:
        given = str(server_id) if server_id else ""
        call_id = given if given and given not in self._taken else minted_call_id(name)
        self._taken.add(call_id)
        return call_id
