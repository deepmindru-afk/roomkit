"""The one log line a failed turn, delivery or task gets (RFC §15.2).

A ``ProviderError`` is not a code defect: it is logged without a traceback, at
the level its status calls for, with the provider and the status in the line
(RFC §15.3). Anything else is unexpected and keeps its full traceback. The
component that raises leaves the line to the one that catches, so an incident
is logged once.
"""

from __future__ import annotations

import logging
from typing import Any

from roomkit.providers.ai.base import ProviderError


def provider_error_level(exc: ProviderError) -> int:
    """ERROR for a missing model or a server fault, WARNING for a transient.

    A 404 (the model does not exist) or a 5xx needs someone to act; no status
    (connect refused, timeout), a 429 or another 4xx is expected now and then.
    """
    status = exc.status_code
    if status == 404 or (status is not None and status >= 500):
        return logging.ERROR
    return logging.WARNING


def log_failure(
    log: logging.Logger,
    exc: Exception,
    what: str,
    *,
    caller_logs: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log that *what* failed with *exc*, once, at the level its cause calls for.

    ``caller_logs`` is for a caller that receives the error itself and owns its
    log line (``InboundResult.error``): a provider error is then only a DEBUG
    line here, so the incident is not reported twice.
    """
    if not isinstance(exc, ProviderError):
        log.exception("%s failed", what, extra=extra)
        return
    level = logging.DEBUG if caller_logs else provider_error_level(exc)
    log.log(
        level,
        "%s failed (provider=%s, status=%s): %s",
        what,
        exc.provider,
        exc.status_code,
        exc,
        extra=extra,
    )
