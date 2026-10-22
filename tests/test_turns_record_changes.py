"""What a turn's channel writes into its record reaches the owner across the
turn backend (RFC §10.6.3, §10.6.4), and costs a copy of the record only when
it may have changed: never one per delta of a stream that leaves it alone.

Each test runs on the in-memory backend, and across a real Redis when
REDIS_URL is set."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.streaming import (
    SegmentBreakMarker,
    ThinkingDeltaMarker,
    ToolCallEndMarker,
    ToolCallStartMarker,
)
from roomkit.turns import TurnClient, TurnExecutor, _executor_changes
from roomkit.turns.base import TurnBackend
from tests.test_turns_backend import ScriptedChannel, _run

# How a context retrieval writes the documents a turn read (≈ 10 KB of JSON).
_SOURCES = [
    {"document_id": f"doc-{i:04d}", "title": f"Document {i}", "excerpt": "x" * 200}
    for i in range(30)
]


@pytest.fixture
def backend(turn_backend: TurnBackend) -> TurnBackend:
    return turn_backend


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


async def _waits_after_a_change_inside_a_value(channel: ScriptedChannel) -> AsyncIterator[Any]:
    channel.record["sources"] = ["a"]
    yield "One, "
    await asyncio.sleep(0)
    channel.record["sources"].append("b")
    yield "two, "
    await asyncio.Event().wait()  # a model that says no more: its reader stops it


async def _fails_after_a_change_inside_a_value(channel: ScriptedChannel) -> AsyncIterator[Any]:
    channel.record["sources"] = ["a"]
    yield "One, "
    await asyncio.sleep(0)
    channel.record["sources"].append("b")
    yield "two, "
    await asyncio.sleep(0)
    raise RuntimeError("the model went away")


@pytest.mark.parametrize("stop", ["close", "drop", "error"])
async def test_a_change_inside_a_value_reaches_the_owner_however_the_stream_stops(
    backend: TurnBackend, stop: str
) -> None:
    """No item that is no delta follows the change: the stream's close, or
    its error, carries it."""
    failing = stop == "error"
    script = (
        _fails_after_a_change_inside_a_value if failing else _waits_after_a_change_inside_a_value
    )
    client = _client(backend, ScriptedChannel(script))
    output = await _run(client)
    stream = output.response_stream
    assert [await anext(stream), await anext(stream)] == ["One, ", "two, "]
    if stop == "close":
        await stream.aclose()
    elif stop == "drop":
        stream.drop()
    else:
        with pytest.raises(RuntimeError):
            await anext(stream)
    async with asyncio.timeout(5):
        while output.response_metadata["sources"] != ["a", "b"]:
            await asyncio.sleep(0.01)
    await client.close()


class _BareRecordChannel(ScriptedChannel):
    """A channel whose output holds a bare dict as its record: ``model_copy``
    skips the validation that would wrap it."""

    def __init__(self, script: Any) -> None:
        super().__init__(script)
        self.bare: dict[str, Any] = {"source": "bare"}

    async def on_event(self, *args: Any) -> Any:
        output = await super().on_event(*args)
        return output.model_copy(update={"response_metadata": self.bare})


async def test_a_record_left_a_bare_dict_still_reaches_the_owner_before_each_item(
    backend: TurnBackend,
) -> None:
    async def script(channel: _BareRecordChannel) -> AsyncIterator[Any]:
        yield "One, "
        await asyncio.sleep(0)
        channel.bare["sources"] = ["a"]
        yield "two."

    client = _client(backend, _BareRecordChannel(script))
    output = await _run(client)
    seen = [
        (_name(item), output.response_metadata.get("sources"))
        async for item in output.response_stream
    ]
    await client.close()

    assert output.response_metadata["source"] == "bare"
    assert seen == [("One, ", None), ("two.", ["a"])]


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
