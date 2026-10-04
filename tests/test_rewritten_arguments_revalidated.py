"""The arguments BEFORE_TOOL_USE leaves are validated again, in one wording,
on every door (RMK-482, RFC §21.1).

A hook may hand back new arguments or edit the event's own in place; either
way the call is refused when they no longer fit the schema, before its
handler runs, and the model reads that the arguments were rewritten: the
model's own call was validated before the hook ran.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import HookResult
from tests.tool_doors import DOORS, Hooks, run_door

SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
REFUSAL = "Invalid rewritten arguments for 'lookup': missing required argument 'city'"


def _edits(how: str) -> Any:
    async def before(event: Any, ctx: Any) -> HookResult:
        if how == "in-place":
            event.arguments.pop("city", None)
            return HookResult.allow()
        return HookResult(action="allow", metadata={"arguments": {}})

    return before


@pytest.mark.parametrize("door", DOORS)
@pytest.mark.parametrize("how", ["in-place", "returned"])
async def test_arguments_a_hook_broke_are_refused_as_rewritten(door: str, how: str) -> None:
    served: list[dict[str, Any]] = []

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        served.append(arguments)
        return "sunny"

    seen = await run_door(
        door, lookup, arguments={"city": "Paris"}, hooks=Hooks(before=_edits(how)), schema=SCHEMA
    )

    # A backend reads (text, is_error, refused); every other door the text.
    read = seen.model_read[0] if isinstance(seen.model_read, tuple) else seen.model_read
    assert REFUSAL in str(read)
    assert served == []
