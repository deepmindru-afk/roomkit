"""A stored result can be searched, and Tool Search switches on for cost (RFC §21.5, RMK-321).

``read_stored_result(result_id, query=...)`` returns the lines that contain
one line of text, case aside, with two lines around each and their numbers,
bounded as a page is: a long line is cut around the match. It searches each
line as stored, so no match says the text is absent. And a catalogue past
``tool_search_threshold_tokens`` hideable schema tokens goes behind Tool
Search whatever the model's window.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from roomkit.channels._stored_read import paginable_lines
from roomkit.channels._tool_eviction import ToolEviction
from roomkit.channels._tool_search import should_activate_tool_search
from roomkit.channels.ai import AIChannel
from roomkit.memory.token_estimator import estimate_tokens, estimate_tool_tokens
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond

_EXPORT = "\n".join(
    f"order A-{n:04d} shipped {'LOST' if n == 1042 else 'ok'} total {n}.00" for n in range(4000)
)


def _stored(eviction: ToolEviction, text: str = _EXPORT) -> str:
    placeholder = eviction.maybe_evict(text, "c0")
    return placeholder.split("'")[1]


def _read(eviction: ToolEviction, **arguments: Any) -> dict[str, Any]:
    return json.loads(eviction.handle_read(arguments))


def test_a_search_returns_the_matching_lines_with_their_neighbours() -> None:
    eviction = ToolEviction(threshold_tokens=500)
    result_id = _stored(eviction)

    found = _read(eviction, result_id=result_id, query="a-1042")

    assert (found["total_matches"], found["has_more"]) == (1, False)
    assert found["content"].splitlines() == [
        f"{n + 1}: order A-{n:04d} shipped {'LOST' if n == 1042 else 'ok'} total {n}.00"
        for n in range(1040, 1045)
    ]


def test_a_search_that_finds_nothing_says_the_text_is_absent() -> None:
    eviction = ToolEviction(threshold_tokens=500)
    result_id = _stored(eviction)

    found = _read(eviction, result_id=result_id, query="A-9999")

    assert (found["total_matches"], found["content"]) == (0, "")
    assert "it is absent" in found["note"]


def test_matches_past_a_page_are_flagged_and_continue_from_next_offset() -> None:
    eviction = ToolEviction(threshold_tokens=500)
    result_id = _stored(eviction)

    first = _read(eviction, result_id=result_id, query="shipped ok")
    rest = _read(eviction, result_id=result_id, query="shipped ok", offset=first["next_offset"])
    past = _read(eviction, result_id=result_id, query="shipped ok", offset=5000)

    assert first["has_more"] and first["total_matches"] == 3999
    assert first["warning"].startswith("PARTIAL CONTENT")
    first_line = first["content"].splitlines()[0].split(":")[0]
    assert rest["content"].splitlines()[0].split(":")[0] != first_line
    assert "past the last match" in past["note"]


def test_a_text_across_the_cut_of_a_long_line_is_found() -> None:
    """One minified JSON line is paged in budget-sized chunks; a search reads
    the line whole, so an id that straddles a chunk boundary is still found."""
    eviction = ToolEviction(threshold_tokens=500)
    ids = [f"A-{n:05d}" for n in range(3000)]
    text = json.dumps([{"id": i, "status": "shipped"} for i in ids], separators=(",", ":"))
    result_id = _stored(eviction, text)
    chunks = paginable_lines(text, eviction._page_budget())
    straddling = [i for i in ids if not any(i in chunk for chunk in chunks)]
    assert straddling  # the chunking does cut some ids in two

    found = _read(eviction, result_id=result_id, query=straddling[0])

    assert found["total_matches"] == 1
    assert straddling[0] in found["content"]


@pytest.mark.parametrize(
    "text",
    [
        "\n".join(f"hit {n} " + "x" * 4000 for n in range(200)),
        json.dumps([{"id": f"A-{n:05d}"} for n in range(12000)], separators=(",", ":")),
    ],
    ids=["long lines", "one JSON line"],
)
def test_a_search_is_bounded_like_a_page(text: str) -> None:
    eviction = ToolEviction()
    result_id = _stored(eviction, text)

    found = eviction.handle_read({"result_id": result_id, "query": "hit 7 "})
    found_json = eviction.handle_read({"result_id": result_id, "query": "A-06000"})

    for answer in (found, found_json):
        assert estimate_tokens(answer) <= eviction.threshold_tokens


def test_a_blank_query_reads_a_page_and_a_query_on_two_lines_is_refused() -> None:
    eviction = ToolEviction(threshold_tokens=500)
    result_id = _stored(eviction)

    assert _read(eviction, result_id=result_id, query="  ")["offset"] == 0
    assert "one line" in _read(eviction, result_id=result_id, query="A-1\nA-2")["error"]


def _export_script() -> list[AIResponse]:
    def call(call_id: str, name: str, **arguments: Any) -> AIResponse:
        return AIResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id=call_id, name=name, arguments=arguments)],
        )

    return [
        call("c0", "export"),
        call("c1", "read_stored_result", result_id="evicted_c0", query="LOST"),
        AIResponse(content="A-1042 was lost.", finish_reason="stop"),
    ]


async def _turn(ch: AIChannel, tools: list[AITool]) -> LoopRun:
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [t.model_dump() for t in tools]},
    )
    event = make_event(body="go", channel_id="sms1")
    return await respond(ch, event, binding, RoomContext(room=Room(id="r1")))


async def test_the_model_searches_a_result_it_evicted(streaming: bool) -> None:
    async def export(name: str, arguments: dict[str, Any]) -> str:
        return _EXPORT

    provider = MockAIProvider(ai_responses=_export_script(), streaming=streaming)
    ch = AIChannel("ai1", provider=provider, tool_handler=export, evict_threshold_tokens=500)

    run = await _turn(ch, [AITool(name="export", description="Export orders", parameters={})])

    found = json.loads(str(run.calls[-1].result))
    assert found["total_matches"] == 1
    assert "1043: order A-1042 shipped LOST" in found["content"]


def _catalogue(count: int) -> list[AITool]:
    props = {f"f{k}": {"type": "string", "description": "A filter, free text."} for k in range(6)}
    return [
        AITool(
            name=f"report_{n}",
            description="Build a report over orders, refunds and shipments by region.",
            parameters={"type": "object", "properties": props},
        )
        for n in range(count)
    ]


class _LargeWindow(MockAIProvider):
    """A mock with a 1M-token window: 10 % of it is 100K schema tokens."""

    @property
    def context_window(self) -> int | None:
        return 1_000_000


def _active(tools: list[AITool], window: int | None, cap: int | None) -> bool:
    return should_activate_tool_search(
        mode=None,
        catalogue=tools,
        pinned=set(),
        window=window,
        threshold_pct=10.0,
        threshold_count=20,
        threshold_tokens=cap,
    )


def test_a_catalogue_past_the_token_cap_goes_behind_tool_search() -> None:
    small, large = _catalogue(4), _catalogue(80)
    assert sum(map(estimate_tool_tokens, small)) < 8000 < sum(map(estimate_tool_tokens, large))

    assert _active(large, 1_000_000, 8000) and not _active(small, 1_000_000, 8000)
    assert not _active(large, 1_000_000, None)
    assert _active(large, None, 8000) and not _active(small, None, 8000)


@pytest.mark.parametrize(("cap", "hidden"), [(8000, True), (None, False)])
async def test_the_channel_hides_a_large_catalogue_on_a_large_window(
    streaming: bool, cap: int | None, hidden: bool
) -> None:
    provider = _LargeWindow(ai_responses=[AIResponse(content="ok")], streaming=streaming)
    ch = AIChannel("ai1", provider=provider, tool_search_threshold_tokens=cap)

    await _turn(ch, _catalogue(80))

    declared = {t.name for t in provider.calls[0].tools if not t.defer_loading}
    assert ("find_tools" in declared) is hidden
    assert ("report_0" in declared) is not hidden


@pytest.mark.parametrize("cap", [0, -1, True, "8000"])
def test_the_token_cap_is_a_positive_number_or_none(cap: Any) -> None:
    with pytest.raises(ValueError, match="tool_search_threshold_tokens"):
        AIChannel("ai1", provider=MockAIProvider(), tool_search_threshold_tokens=cap)
