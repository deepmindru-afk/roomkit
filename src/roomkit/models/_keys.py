"""An idempotency key as the models take it (RFC §13.4)."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator


def _no_blank_key(key: str | None) -> str | None:
    """*key*, or None when it is blank: a blank key names no message, so the
    pipeline looks none up and no store indexes one (RFC §13.4)."""
    if key is not None and not key.strip():
        return None
    return key


IdempotencyKey = Annotated[str | None, AfterValidator(_no_blank_key)]
"""An optional idempotency key, a blank one read as none."""
