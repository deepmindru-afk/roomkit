"""How a turn's stream is pulled across the turn backend (RFC §10.6.4): the
executor reads deltas ahead of the owner's reader, never past any other item
before the reader read it; a cut closes the channel's stream wherever it
finds the stream; text bursts cross whole, or refuse the turn."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit.core._turn_scope import current_turn_id
from roomkit.core.exceptions import TurnLinkError
from roomkit.models.steering import Cancel
from roomkit.models.streaming import SegmentBreakMarker, ToolCallStartMarker
from roomkit.turns import InMemoryTurnBackend, TurnClient, TurnExecutor, memory
from roomkit.turns._frame_sender import FrameSender
from roomkit.turns.wire import Item
from tests.test_turns_backend import ScriptedChannel, _client, _run


@pytest.mark.parametrize("deltas", [10, 40])
async def test_a_call_never_runs_before_its_reader_read_its_start(deltas: int) -> None:
    """Past half its window the owner pulls again while the executor may
    already hold the call's start: that pull grants nothing past it."""
    ran: list[str] = []

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        for i in range(deltas):
            yield f"w{i} "
        yield ToolCallStartMarker(tool_name="t", tool_id="c1", arguments={})
        ran.append("the call ran")
        yield "after"

    client = _client(ScriptedChannel(script))
    stream = (await _run(client)).response_stream
    async for item in stream:
        if isinstance(item, ToolCallStartMarker):
            break  # the reader stops at the call
    await asyncio.sleep(0.05)
    await stream.aclose()
    await client.close()

    assert ran == []


def _closing_script(log: list[str]) -> Any:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        try:
            yield "text"
            yield SegmentBreakMarker()  # past it, the executor waits for a pull
            yield "never"
        finally:
            log.append(f"closed under {current_turn_id()}")

    return script


async def test_a_cut_while_the_executor_waits_for_a_pull_closes_the_stream_in_the_turn() -> None:
    """A steering Cancel to a channel without steering cuts the turn: its
    stream is closed under the turn, not left to the garbage collector."""
    log: list[str] = []
    client = _client(ScriptedChannel(_closing_script(log)))
    output = await _run(client)
    stream = output.response_stream
    assert await anext(stream) == "text"
    assert isinstance(await anext(stream), SegmentBreakMarker)
    await asyncio.sleep(0.05)

    assert await client.steer(Cancel(), "agent", room_id="room") == 1
    assert [item async for item in stream] == []
    turn_id = client._started[0].request.turn_id
    await client.close()

    assert log == [f"closed under {turn_id}"]


async def test_an_owner_lost_while_the_executor_waits_for_a_pull_closes_the_stream() -> None:
    log: list[str] = []
    channels = {"agent": ScriptedChannel(_closing_script(log))}
    backend = InMemoryTurnBackend()
    executor = TurnExecutor(channels.get, hosts=channels.__contains__)
    await backend.serve(executor)
    client = TurnClient(backend)
    stream = (await _run(client)).response_stream
    assert await anext(stream) == "text"
    assert isinstance(await anext(stream), SegmentBreakMarker)
    await asyncio.sleep(0.05)

    await client.close()  # the owner goes
    for _ in range(50):
        if log:
            break
        await asyncio.sleep(0.01)
    await executor.close()

    assert len(log) == 1 and log[0].startswith("closed under turn-")


async def test_a_text_delta_no_frame_can_carry_refuses_the_turn() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "Hello "
        await asyncio.sleep(0.01)
        yield "bad \ud800 text "
        yield "world."

    client = _client(ScriptedChannel(script))
    stream = (await _run(client)).response_stream
    read: list[str] = []
    with pytest.raises(TypeError, match="stream item cannot cross"):
        async for item in stream:
            read.append(item)
    await client.close()

    assert read == ["Hello "]


async def test_text_kept_while_a_send_waits_crosses_without_another_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over a link whose send waits (a network write), a delta kept while
    the burst before it was being sent goes as soon as that send ends."""
    send = memory._ExecutorLink.send

    async def network_send(self: Any, frame: Any) -> int:
        await asyncio.sleep(0.005)
        return await send(self, frame)

    monkeypatch.setattr(memory._ExecutorLink, "send", network_send)

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        for i in range(5):
            yield f"t{i} "
            await asyncio.sleep(0.002)
        await asyncio.Event().wait()  # then the model thinks

    client = _client(ScriptedChannel(script))
    stream = (await _run(client)).response_stream
    started, text = time.monotonic(), ""
    while "t4" not in text:
        text += await asyncio.wait_for(anext(stream), 1.0)
    took = time.monotonic() - started
    await stream.aclose()
    await client.close()

    assert text == "t0 t1 t2 t3 t4 "
    assert took < 1.0  # not carried by the next heartbeat


async def test_the_record_a_reader_reads_is_the_one_sent_before_its_item() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "a"
        channel.record["x"] = 1
        yield "b"
        yield "b2"
        yield SegmentBreakMarker()
        channel.record["x"] = 2
        yield "c"
        channel.record["x"] = 3

    client = _client(ScriptedChannel(script))
    output = await _run(client)
    seen = []
    async for item in output.response_stream:
        await asyncio.sleep(0.02)  # a slow reader: the turn's end comes first
        name = item if isinstance(item, str) else type(item).__name__
        seen.append((name, output.response_metadata.get("x")))
    seen.append(("end", output.response_metadata.get("x")))
    await client.close()

    assert seen == [
        ("a", None),
        ("bb2", 1),
        ("SegmentBreakMarker", 1),
        ("c", 2),
        ("end", 3),
    ]


async def test_a_burst_naming_no_positive_number_of_deltas_is_an_invalid_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def foreign_burst(self: FrameSender) -> None:
        if self._text:
            texts, self._text = self._text, []
            await self._send(Item(item={"kind": "text", "text": "".join(texts), "deltas": "two"}))

    monkeypatch.setattr(FrameSender, "_send_text", foreign_burst)

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "a"
        yield "b"
        await asyncio.Event().wait()

    client = _client(ScriptedChannel(script))
    stream = (await _run(client)).response_stream
    with pytest.raises(TurnLinkError) as raised:
        await anext(stream)
    [turn] = client._started

    assert raised.value.reason == "frame_invalid"
    assert turn.over  # the turn ends, its reader told why
    await client.close()
