"""What a provider sends: its tools declared, and earlier rounds replayed as
its wire needs them (RFC §6.4, §6.7)."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.providers.ai.base import (
    AIImagePart,
    AIMessage,
    AITextPart,
    AIThinkingPart,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
    ProviderError,
    thinking_parts_of,
)
from roomkit.providers.ai.round_parts import round_parts
from tests.text_conformance.driver import (
    IMAGE_RESULTS,
    REDACTED_REASONING,
    SCHEMA_AS_GIVEN,
    Driver,
)
from tests.text_conformance.scenario import LOOKUP, PNG, generation, tool_context
from tests.text_conformance.script import Call, Reasoning, Script


class TestDeclaration:
    async def test_a_tool_without_parameters_is_declared_as_an_empty_object(
        self, driver: Driver, mode: str
    ) -> None:
        bare = AITool(name="now", description="Current time.")
        await generation(driver, Script(text="ok"), mode, tool_context(bare))

        assert driver.declared(driver.requests[0])["now"] == {
            "type": "object",
            "properties": {},
        }

    async def test_a_schema_is_declared_as_given(self, driver: Driver) -> None:
        driver.require(SCHEMA_AS_GIVEN)
        schema = {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "enum": [1, 2, 3]},
                "place": {"$ref": "#/$defs/place"},
            },
            "$defs": {"place": {"type": "string"}},
        }
        tool = AITool(name="set_level", description="d", parameters=schema)
        await generation(driver, Script(text="ok"), "stream", tool_context(tool))

        assert driver.declared(driver.requests[0])["set_level"] == schema

    async def test_a_name_the_vendor_refuses_fails_before_the_request(
        self, driver: Driver
    ) -> None:
        for name in driver.refused_names:
            tool = AITool(name=name, description="d")
            with pytest.raises(ProviderError, match=name):
                await generation(driver, Script(text="ok"), "stream", tool_context(tool))
        assert driver.requests == []
        for name in driver.accepted_names:
            await generation(
                driver,
                Script(text="ok"),
                "stream",
                tool_context(AITool(name=name, description="d")),
            )
            assert name in driver.declared(driver.requests[-1])


def _round(*parts: Any) -> AIMessage:
    return AIMessage(role="assistant", content=list(parts))


def _results(*parts: AIToolResultPart) -> AIMessage:
    return AIMessage(role="tool", content=list(parts))


_CALL = AIToolCallPart(id="c1", name="lookup", arguments={"q": "a"})


async def _replay(driver: Driver, *history: AIMessage) -> list[Any]:
    messages = [AIMessage(role="user", content="go"), *history]
    await generation(driver, Script(text="ok"), "stream", tool_context(LOOKUP, messages=messages))
    return driver.replayed(driver.requests[-1])


class TestReplay:
    async def test_a_call_and_its_results_go_back_paired(self, driver: Driver) -> None:
        second = AIToolCallPart(id="c2", name="lookup", arguments={"q": "b"})
        items = await _replay(
            driver,
            _round(AITextPart(text="Looking."), _CALL, second),
            _results(
                AIToolResultPart(tool_call_id="c1", name="lookup", result="found"),
                AIToolResultPart(tool_call_id="c2", name="lookup", result="denied", is_error=True),
            ),
        )

        calls = [i for i in items if i[0] == "call"]
        results = [i for i in items if i[0] == "result"]
        assert [c[2] for c in calls] == [{"q": "a"}, {"q": "b"}]
        assert [r[2] for r in results] == ["found", "denied"]
        assert [r[1] for r in results] == [c[1] for c in calls]
        flags = [r[3] for r in results]
        assert flags == ([False, True] if driver.error_flag else [None, None])

    async def test_an_image_result_reaches_the_model(self, driver: Driver) -> None:
        driver.require(IMAGE_RESULTS)
        items = await _replay(
            driver,
            _round(_CALL),
            _results(
                AIToolResultPart(
                    tool_call_id="c1",
                    name="lookup",
                    result=[AITextPart(text="shot"), AIImagePart(url=PNG)],
                )
            ),
        )

        assert ("image",) in items

    async def test_reasoning_goes_back_by_the_wires_convention(self, driver: Driver) -> None:
        items = await _replay(
            driver,
            _round(AIThinkingPart(thinking="why", signature="S0"), _CALL),
            _results(AIToolResultPart(tool_call_id="c1", name="lookup", result="found")),
        )

        reasoning = [i for i in items if i[0] in ("thinking", "redacted", "inline", "field")]
        expected = {
            "blocks": [("thinking", "why", "S0")],
            "call_signature": [],
            "inline": [("inline", "why")],
            "field": [("field", "why")],
            "dropped": [],
        }[driver.reasoning]
        assert reasoning == expected

    async def test_a_redacted_block_goes_back_as_received(self, driver: Driver) -> None:
        driver.require(REDACTED_REASONING)
        items = await _replay(
            driver,
            _round(AIThinkingPart(thinking="", redacted="RRR"), _CALL),
            _results(AIToolResultPart(tool_call_id="c1", name="lookup", result="found")),
        )

        assert ("redacted", "RRR") in items


class TestRoundTrip:
    async def test_a_signed_round_goes_back_signed(self, driver: Driver) -> None:
        """A round received, kept as the loop keeps it, then replayed."""
        if driver.reasoning not in ("blocks", "call_signature"):
            pytest.skip(f"{driver.label}: its reasoning goes back unsigned")
        script = Script(
            reasoning=(Reasoning("why", signature="S0"),),
            calls=(
                Call("lookup", '{"q": "a"}', id="c1", index=0),
                Call("lookup", '{"q": "b"}', id="c2", index=1),
            ),
            finish="tool",
        )
        response = await driver.provider(script).generate(tool_context(LOOKUP))
        kept = round_parts(thinking_parts_of(response), response.content, response.tool_calls)

        items = await _replay(
            driver,
            AIMessage(role="assistant", content=kept),
            _results(
                *(
                    AIToolResultPart(tool_call_id=c.id, name=c.name, result="found")
                    for c in response.tool_calls
                )
            ),
        )

        if driver.reasoning == "blocks":
            assert ("thinking", "why", "S0") in items
        else:
            # Gemini signs the first call only; every call goes back signed.
            assert [i[2] for i in items if i[0] == "signature"] == ["S0", "S0"]
