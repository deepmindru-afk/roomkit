"""Provider-reported submission boundary for realtime text injection."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class VoiceInjectionResult(BaseModel):
    """Report what the provider knows after ``inject_text``.

    ``sent`` means its send operation completed, not that audio was heard.
    ``not_sent`` guarantees no submission or pending submission exists.
    ``unknown`` covers acceptance that cannot be established, including input
    queued locally for later sending. Only ``not_sent`` may allow retry.
    Exceptions and ``None`` returns are conservatively unknown.
    """

    status: Literal["sent", "not_sent", "unknown"]
    reason: str | None = None
    retryable: bool = False
