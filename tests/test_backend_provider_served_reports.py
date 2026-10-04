"""A call a reasoning backend's own provider served is reported, as on an
AIChannel (RMK-480, RFC §9.3, §12.4.1).

The provider ran the call itself (``AIToolCall.served``): the voice channel's
ON_TOOL_CALL hooks hear it once through ``ReasoningRequest.report_call``,
served or failed, with what the provider returned.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.providers.ai.base import AIToolCall, ServedCall
from tests.tool_doors import run_door


async def _unused(name: str, arguments: dict[str, Any]) -> str:
    raise AssertionError("the provider served the call")


@pytest.mark.parametrize("door", ["text-stream", "text-nostream", "rt-agent-backend"])
@pytest.mark.parametrize(
    ("served", "expected"),
    [
        (ServedCall(result="3 hits"), (False, "3 hits")),
        (ServedCall(result="quota", is_error=True), (True, "quota")),
    ],
    ids=["served", "failed"],
)
async def test_a_call_the_provider_served_is_reported_once(
    door: str, served: ServedCall, expected: tuple[bool, str]
) -> None:
    call = AIToolCall(id="c1", name="web_search", arguments={"q": "x"}, served=served)
    seen = await run_door(door, _unused, call=call)

    assert [(e.name, e.is_error, e.result) for e in seen.reports] == [("web_search", *expected)]
