"""ToolEviction per-room scoping.

The eviction buffer lives on a channel object shared by every room the
channel serves — without room scoping, one conversation's oversized tool
output is readable from another conversation, and the re-read tool is
injected into rooms that evicted nothing. A stored result also stays
readable while its room works, under an id nothing else is given
(RMK-285, RFC §21.5).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager

from roomkit.channels._tool_eviction import ToolEviction
from roomkit.channels._tool_usage import ToolUsageMemory
from roomkit.channels.ai import _current_loop_ctx, _ToolLoopContext

_BIG = "line\n" * 20_000  # far past the default 5000-token threshold


def _in_room(room_id: str):
    return _current_loop_ctx.set(_ToolLoopContext(room_id=room_id))


class TestRoomScope:
    def test_read_is_scoped_to_the_evicting_room(self) -> None:
        ev = ToolEviction()

        token = _in_room("room-a")
        try:
            ev.maybe_evict(_BIG, "tc1")
        finally:
            _current_loop_ctx.reset(token)

        token = _in_room("room-b")
        try:
            out = json.loads(ev.handle_read({"result_id": "evicted_tc1"}))
        finally:
            _current_loop_ctx.reset(token)

        assert "error" in out
        # The other room's ids must not leak through the error hint either.
        assert out["available"] == []

        token = _in_room("room-a")
        try:
            out = json.loads(ev.handle_read({"result_id": "evicted_tc1"}))
        finally:
            _current_loop_ctx.reset(token)
        assert "content" in out

    def test_has_evicted_is_per_room(self) -> None:
        ev = ToolEviction()

        token = _in_room("room-a")
        try:
            ev.maybe_evict(_BIG, "tc1")
            assert ev.has_evicted
        finally:
            _current_loop_ctx.reset(token)

        token = _in_room("room-b")
        try:
            assert not ev.has_evicted
        finally:
            _current_loop_ctx.reset(token)

    def test_fallback_scope_outside_tool_loop(self) -> None:
        ev = ToolEviction()
        ev.maybe_evict(_BIG, "tc1")
        assert ev.has_evicted
        out = json.loads(ev.handle_read({"result_id": "evicted_tc1"}))
        assert "content" in out


_SMALL = "line\n" * 200


@contextmanager
def _room(room_id: str) -> Iterator[None]:
    token = _in_room(room_id)
    try:
        yield
    finally:
        _current_loop_ctx.reset(token)


def _evict(ev: ToolEviction, room: str, text: str = _SMALL, call_id: str = "c") -> str:
    """Evict *text* in *room*; return the id its placeholder names."""
    with _room(room):
        return ev.maybe_evict(text, call_id).split("'")[1]


def _read(ev: ToolEviction, room: str, result_id: str) -> dict:
    with _room(room):
        return json.loads(ev.handle_read({"result_id": result_id}))


class TestCapacity:
    def test_other_rooms_evictions_do_not_push_a_rooms_result_out(self) -> None:
        ev = ToolEviction(threshold_tokens=10)
        kept = _evict(ev, "room-a", call_id="call_0")
        for i in range(60):
            _evict(ev, f"room-{i}", call_id=f"c{i}")

        assert "error" not in _read(ev, "room-a", kept)

    def test_a_quiet_room_keeps_its_result_while_full_rooms_evict(self) -> None:
        ev = ToolEviction(threshold_tokens=10)
        kept = _evict(ev, "quiet")
        for room in ("w", "x", "y", "z"):
            for i in range(50):
                _evict(ev, room, call_id=f"c{i}")

        assert "error" not in _read(ev, "quiet", kept)

    def test_a_room_keeps_its_fifty_most_recently_read(self) -> None:
        ev = ToolEviction(threshold_tokens=10)
        ids = [_evict(ev, "room-a", call_id=f"c{i}") for i in range(50)]
        assert "error" not in _read(ev, "room-a", ids[0])  # read back: in use

        _evict(ev, "room-a", call_id="c50")

        assert "error" not in _read(ev, "room-a", ids[0])
        assert "error" in _read(ev, "room-a", ids[1])

    def test_the_whole_store_stays_bounded(self) -> None:
        ev = ToolEviction(threshold_tokens=10)
        ids = [_evict(ev, f"room-{i}") for i in range(300)]

        readable = [i for i, rid in enumerate(ids) if "error" not in _read(ev, f"room-{i}", rid)]
        assert len(readable) == 200


class TestIds:
    def test_a_call_id_reused_in_a_later_turn_overwrites_nothing(self) -> None:
        ev = ToolEviction(threshold_tokens=400)
        first = _evict(ev, "room-b", "FIRST TURN DATA\n" * 100, "call_0")
        second = _evict(ev, "room-b", "SECOND TURN DATA\n" * 100, "call_0")

        assert first != second
        assert _read(ev, "room-b", first)["content"].startswith("FIRST TURN DATA")
        assert _read(ev, "room-b", second)["content"].startswith("SECOND TURN DATA")

    def test_an_id_that_left_the_store_is_not_given_again(self) -> None:
        ev = ToolEviction(threshold_tokens=10)
        old = _evict(ev, "room-a", "ROUND ONE\n" * 100, "call_0")
        for i in range(60):
            _evict(ev, "room-a", call_id=f"c{i}")  # pushes call_0's result out

        new = _evict(ev, "room-a", "ROUND TWO\n" * 100, "call_0")

        assert new != old
        assert "error" in _read(ev, "room-a", old)


def test_a_rebuilt_digest_offers_no_stored_id() -> None:
    placeholder = ToolEviction(threshold_tokens=50).maybe_evict("row\n" * 2000, "toolu_1")
    memory = ToolUsageMemory()

    memory.seed("r1", [{"name": "list_cards", "arguments": {"board": "x"}, "result": placeholder}])

    digest = memory.render_digest("r1") or ""
    assert "evicted_" not in digest
    assert "read_stored_result" not in digest
    assert "Result too large (2001 tokens), not kept" in digest
