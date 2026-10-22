"""The keys a test wrote in the Redis REDIS_URL names, under a prefix of its
own: deleted once it is done, and only those."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest


def key_prefix() -> str:
    """A key prefix of one test's own."""
    return f"roomkit:test:{uuid4().hex[:12]}"


async def delete_keys(prefix: str) -> None:
    """Delete the keys under *prefix* in the Redis REDIS_URL names."""
    redis = pytest.importorskip("redis.asyncio")
    client = redis.from_url(os.environ["REDIS_URL"])
    keys = [key async for key in client.scan_iter(f"{prefix}:*")]
    if keys:
        await client.delete(*keys)
    await client.aclose()
