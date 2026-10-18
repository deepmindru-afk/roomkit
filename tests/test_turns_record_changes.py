"""What a turn's channel writes into its record reaches the owner across the
turn backend (RFC §10.6.3, §10.6.4), and costs a copy of the record only when
it may have changed: never one per delta of a stream that leaves it alone.

Each test runs on the in-memory backend, and across a real Redis when
REDIS_URL is set."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4

import pytest

from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.streaming import (
    SegmentBreakMarker,
    ThinkingDeltaMarker,
    ToolCallEndMarker,
    ToolCallStartMarker,
)
from roomkit.turns import InMemoryTurnBackend, TurnClient, TurnExecutor, _executor_changes
from roomkit.turns.base import TurnBackend
from roomkit.turns.redis import RedisTurnBackend
from tests.test_turns_backend import ScriptedChannel, _run

# How a context retrieval writes the documents a turn read (≈ 10 KB of JSON).
_SOURCES = [
    {"document_id": f"doc-{i:04d}", "title": f"Document {i}", "excerpt": "x" * 200}
    for i in range(30)
]


@pytest.fixture(params=["memory", "redis"])
async def backend(request: pytest.FixtureRequest) -> AsyncIterator[TurnBackend]:
    if request.param == "memory":
        yield InMemoryTurnBackend()
        return
    if not os.environ.get("REDIS_URL"):
        pytest.skip("REDIS_URL not set")
    redis = pytest.importorskip("redis.asyncio")
    prefix = f"roomkit:test:{uuid4().hex[:12]}"
    # The owner reaches its own executor across Redis, as one elsewhere.
    yield RedisTurnBackend(os.environ["REDIS_URL"], key_prefix=prefix, _loop_back=True)
    client = redis.from_url(os.environ["REDIS_URL"])
    keys = [key async for key in client.scan_iter(f"{prefix}:*")]
    if keys:
        await client.delete(*keys)
    await client.aclose()


def _client(backend: TurnBackend, channel: ScriptedChannel) -> TurnClient:
    return TurnClient(
        backend,
        executor=TurnExecutor({channel.channel_id: channel}.get, hosts={channel.channel_id}),
    )


def _name(item: Any) -> str:
    return item if isinstance(item, str) else type(item).__name__


async def _read_with_records(
    backend: TurnBackend, script: Any
) -> list[tuple[str, list[str] | None]]:
    """Each item the owner reads, with the record's ``sources`` as it reads it."""
    client = _client(backend, ScriptedChannel(script))
    output = await _run(client)
    seen = []
    async for item in output.response_stream:
        sources = output.response_metadata.get("sources")
        seen.append((_name(item), None if sources is None else list(sources)))
    await client.close()
    return seen


async def test_a_change_before_a_call_s_start_or_end_reaches_the_owner_before_it(
    backend: TurnBackend,
) -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "Looking it up."
        channel.record["sources"] = ["a"]
        channel.record["sources"].append("b")
        yield ToolCallStartMarker(tool_name="search", tool_id="c1", arguments={})
        # Inside a value the record holds: none of its keys is written.
        channel.record["sources"].append("c")
        yield ToolCallEndMarker(tool_name="search", tool_id="c1", result="found")
        yield "Found it."

    assert await _read_with_records(backend, script) == [
        ("Looking it up.", None),
        ("ToolCallStartMarker", ["a", "b"]),
        ("ToolCallEndMarker", ["a", "b", "c"]),
        ("Found it.", ["a", "b", "c"]),
    ]


async def test_a_change_inside_a_value_between_deltas_reaches_the_owner_before_its_segment_ends(
    backend: TurnBackend,
) -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        channel.record["sources"] = ["a"]
        yield "One, "
        await asyncio.sleep(0)  # a frame per delta, as a model read off the network
        channel.record["sources"].append("b")
        yield "two."
        yield SegmentBreakMarker()

    seen = await _read_with_records(backend, script)

    assert seen[0] == ("One, ", ["a"])
    assert seen[-1] == ("SegmentBreakMarker", ["a", "b"])


def _streams(deltas: int, delta: Callable[[int], Any]) -> Any:
    """A stream writing its record before its first delta and before its
    last, a frame per delta between them."""

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        channel.record["sources"] = _SOURCES
        for i in range(deltas):
            yield delta(i)
            await asyncio.sleep(0)
        channel.record["usage"] = {"deltas": deltas}
        yield delta(deltas)

    return script


@pytest.mark.parametrize(
    "delta",
    [lambda i: f"w{i} ", lambda i: ThinkingDeltaMarker(thinking=f"t{i} ")],
    ids=["text", "reasoning"],
)
async def test_a_turn_copies_its_record_no_more_often_for_more_deltas(
    backend: TurnBackend, delta: Callable[[int], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    copies: list[ResponseMetadata] = []
    copy = _executor_changes.record_payload

    def counted(record: ResponseMetadata) -> dict[str, Any]:
        copies.append(record)
        return copy(record)

    monkeypatch.setattr(_executor_changes, "record_payload", counted)
    channel = ScriptedChannel()
    client = _client(backend, channel)
    counts = []
    for deltas in (4, 40):
        channel.script, channel.record = _streams(deltas, delta), ResponseMetadata()
        output = await _run(client)
        read = [item async for item in output.response_stream]
        assert len(read) == deltas + 1
        assert output.response_metadata == {"sources": _SOURCES, "usage": {"deltas": deltas}}
        counts.append(len(copies))
        copies.clear()
    await client.close()

    assert counts[0] > 0
    assert counts[1] == counts[0]
