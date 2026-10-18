"""The suite with every kit's turns across a real Redis (RFC §10.6).

``ROOMKIT_TEST_TURN_BACKEND=redis REDIS_URL=redis://... uv run pytest`` gives
each kit built with no turn backend a RedisTurnBackend whose own executor is
reached across Redis too, in a fresh context, as an executor in another
process would be: the intelligence channels' tests then run through the
distributed transport. A test that assumes the kit's turns run in its
process is marked ``in_process`` (naming why) and skipped in this mode.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

import roomkit.turns.owner as turns_owner
from roomkit.turns.redis import RedisTurnBackend

ENABLED = os.environ.get("ROOMKIT_TEST_TURN_BACKEND") == "redis"


def _backend() -> RedisTurnBackend:
    return RedisTurnBackend(
        os.environ["REDIS_URL"], key_prefix=f"roomkit:test:{uuid4().hex[:12]}", _loop_back=True
    )


def install(config: pytest.Config) -> None:
    """Give every kit built with no turn backend one across Redis."""
    if not os.environ.get("REDIS_URL"):
        raise pytest.UsageError("ROOMKIT_TEST_TURN_BACKEND=redis needs REDIS_URL")
    turns_owner.InMemoryTurnBackend = _backend  # type: ignore[misc, assignment]
    # A test that never closes its kit leaves its backend's connection pool to
    # the collector, whose release redis-py never awaits: noise of this mode.
    config.addinivalue_line(
        "filterwarnings", "ignore:coroutine 'ConnectionPool.release' was never awaited"
    )
