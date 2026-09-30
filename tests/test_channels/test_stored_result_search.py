"""A stored result can be searched, and Tool Search switches on for cost (RFC §21.5, RMK-321).

``read_stored_result(result_id, query=...)`` returns the lines that contain
the text, case aside, with two lines around each and their numbers, bounded
as a page is. It covers the whole result, so no match says the text is
absent. And a catalogue past ``tool_search_threshold_tokens`` deferrable
schema tokens goes behind Tool Search whatever the model's window.
"""

from __future__ import annotations

import json
from typing import Any

from roomkit.channels._tool_eviction import ToolEviction
from roomkit.channels._tool_search import should_activate_tool_search
from roomkit.channels.ai import AIChannel
from roomkit.memory.token_estimator import estimate_tool_tokens
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import respond

_EXPORT = "\n".join(
    f"order A-{n:04d} shipped {'LOST' if n == 1042 else 'ok'} total {n}.00" for n in range(4000)
)


def _stored(eviction: ToolEviction) -> str:
    placeholder = eviction.maybe_evict(_EXPORT, "c0")
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
    assert "covered the whole result" in found["note"]


def test_matches_past_a_page_continue_from_next_offset() -> None:
    eviction = ToolEviction(threshold_tokens=500)
    result_id = _stored(eviction)

    first = _read(eviction, result_id=result_id, query="shipped ok")
    rest = _read(eviction, result_id=result_id, query="shipped ok", offset=first["next_offset"])

    assert first["has_more"] and first["total_matches"] == 3999
    assert len(first["content"]) <= eviction._page_budget()
    assert (
        rest["content"].splitlines()[0].split(":")[0]
        != first["content"].splitlines()[0].split(":")[0]
    )


def test_an_empty_query_is_refused() -> None:
    eviction = ToolEviction(threshold_tokens=500)

    assert "error" in _read(eviction, result_id=_stored(eviction), query="  ")


async def test_the_model_searches_a_result_it_evicted(streaming: bool) -> None:
    lookup = AITool(name="export", description="Export orders", parameters={})
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c0", name="export", arguments={})],
            ),
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    AIToolCall(
                        id="c1",
                        name="read_stored_result",
                        arguments={"result_id": "evicted_c0", "query": "LOST"},
                    )
                ],
            ),
            AIResponse(content="A-1042 was lost.", finish_reason="stop"),
        ],
        streaming=streaming,
    )

    async def export(name: str, arguments: dict[str, Any]) -> str:
        return _EXPORT

    ch = AIChannel("ai1", provider=provider, tool_handler=export, evict_threshold_tokens=500)
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [lookup.model_dump()]},
    )

    run = await respond(
        ch,
        make_event(body="which was lost?", channel_id="sms1"),
        binding,
        RoomContext(room=Room(id="r1")),
    )

    found = json.loads(str(run.calls[-1].result))
    assert found["total_matches"] == 1
    assert "1043: order A-1042 shipped LOST" in found["content"]


def _catalogue(count: int) -> list[AITool]:
    props = {
        f"f{k}": {"type": "string", "description": "A filter, as free text."} for k in range(6)
    }
    return [
        AITool(
            name=f"report_{n}",
            description="Build a report over orders, refunds and shipments by region.",
            parameters={"type": "object", "properties": props},
        )
        for n in range(count)
    ]


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

    # A 1M window puts the fit threshold at 100K tokens: the cap decides.
    assert _active(large, 1_000_000, 8000) and not _active(small, 1_000_000, 8000)
    assert not _active(large, 1_000_000, None)
    # A window unknown: the cap, or the tool count.
    assert _active(large, None, 8000) and not _active(small, None, 8000)


def test_the_channel_caps_tool_search_at_8000_tokens_by_default() -> None:
    ch = AIChannel("ai1", provider=MockAIProvider())

    assert ch._tool_search_threshold_tokens == 8000
