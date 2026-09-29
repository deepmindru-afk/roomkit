"""A stored large result stays readable while its room works (RMK-285, RFC §21.5).

The store's capacity is per room, each eviction has its own id, and a digest
rebuilt from persisted history offers no id the store no longer holds.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager

from roomkit.channels._tool_eviction import ToolEviction
from roomkit.channels._tool_usage import ToolUsageMemory
from roomkit.channels.ai import _current_loop_ctx, _ToolLoopContext

_BIG = "line\n" * 200


@contextmanager
def _in_room(room: str) -> Iterator[None]:
    ctx = _ToolLoopContext()
    ctx.room_id = room
    token = _current_loop_ctx.set(ctx)
    try:
        yield
    finally:
        _current_loop_ctx.reset(token)


def _stored_id(placeholder: str) -> str:
    return placeholder.split("'")[1]


def _readable(ev: ToolEviction, result_id: str) -> bool:
    return "error" not in json.loads(ev.handle_read({"result_id": result_id}))


class TestCapacityPerRoom:
    def test_other_rooms_evictions_do_not_push_a_rooms_result_out(self) -> None:
        ev = ToolEviction(threshold_tokens=10)
        with _in_room("room-A"):
            kept = _stored_id(ev.maybe_evict(_BIG, "call_0"))
        for i in range(60):
            with _in_room(f"room-{i}"):
                ev.maybe_evict(_BIG, f"c{i}")

        with _in_room("room-A"):
            assert _readable(ev, kept)

    def test_a_room_keeps_its_fifty_latest(self) -> None:
        ev = ToolEviction(threshold_tokens=10)
        with _in_room("room-A"):
            ids = [_stored_id(ev.maybe_evict(_BIG, f"c{i}")) for i in range(51)]

            assert not _readable(ev, ids[0])
            assert all(_readable(ev, i) for i in ids[1:])

    def test_every_room_together_stays_bounded(self) -> None:
        ev = ToolEviction(threshold_tokens=10)
        for i in range(300):
            with _in_room(f"room-{i}"):
                ev.maybe_evict(_BIG, "c")

        assert len(ev._store) == 200


def test_a_call_id_reused_in_a_later_turn_overwrites_nothing() -> None:
    ev = ToolEviction(threshold_tokens=400)
    with _in_room("room-B"):
        first = _stored_id(ev.maybe_evict("FIRST TURN DATA\n" * 100, "call_0"))
        second = _stored_id(ev.maybe_evict("SECOND TURN DATA\n" * 100, "call_0"))

        assert first != second
        pages = [json.loads(ev.handle_read({"result_id": i}))["content"] for i in (first, second)]
    assert pages[0].startswith("FIRST TURN DATA")
    assert pages[1].startswith("SECOND TURN DATA")


def test_a_rebuilt_digest_offers_no_stored_id() -> None:
    placeholder = ToolEviction(threshold_tokens=50).maybe_evict("row\n" * 2000, "toolu_1")
    memory = ToolUsageMemory()

    memory.seed("r1", [{"name": "list_cards", "arguments": {"board": "x"}, "result": placeholder}])

    digest = memory.render_digest("r1") or ""
    assert "evicted_" not in digest
    assert "read_stored_result" not in digest
    assert "Result too large (2001 tokens)" in digest
