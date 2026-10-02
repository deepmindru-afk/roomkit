"""What a provider hands the loop from a response: its calls, whole or not
runnable, its reasoning and its usage (RFC §6.4)."""

from __future__ import annotations

import pytest

from tests.text_conformance.driver import (
    ARGUMENT_TEXT,
    CACHE_USAGE,
    CALL_INDEX,
    CALLS_IN_ONE_CHUNK,
    COMPOSITION,
    REASONING_USAGE,
    REDACTED_REASONING,
    REPEATED_ID,
    RESPONSE_CALL_WITHOUT_ID,
    SIGNED_REASONING,
    STREAM_USAGE,
    STREAM_WITHOUT_FINISH,
    WRITTEN_UNREADABLE,
    Driver,
)
from tests.text_conformance.scenario import LOOKUP, generation, tool_context
from tests.text_conformance.script import Call, Reasoning, Script, Usage


class TestCalls:
    async def test_a_call_reaches_the_loop_whole(self, driver: Driver, mode: str) -> None:
        script = Script(
            calls=(Call("lookup", '{"q": "paris"}', id="c1", index=0, fragments=3),),
            finish="tool",
        )

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        [call] = answer.calls
        assert (call.name, call.arguments, call.partial) == ("lookup", {"q": "paris"}, False)
        assert call.id

    async def test_two_calls_stay_two_each_with_its_own_id(
        self, driver: Driver, mode: str
    ) -> None:
        script = Script(
            calls=(
                Call("lookup", '{"q": "a"}', id="c1", index=0, fragments=2),
                Call("lookup", '{"q": "b"}', id="c2", index=1, fragments=2),
            ),
            finish="tool",
        )

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert [c.arguments for c in answer.calls] == [{"q": "a"}, {"q": "b"}]
        assert len({c.id for c in answer.calls}) == 2

    async def test_calls_starting_in_one_chunk_stay_apart(self, driver: Driver) -> None:
        driver.require(CALLS_IN_ONE_CHUNK)
        script = Script(
            calls=(
                Call("lookup", '{"q": "a"}', id="c1", index=0, fragments=2),
                Call("lookup", '{"q": "b"}', id="c2", index=1, fragments=2),
            ),
            finish="tool",
            calls_in_one_chunk=True,
        )

        answer = await generation(driver, script, "stream", tool_context(LOOKUP))

        assert [c.arguments for c in answer.calls] == [{"q": "a"}, {"q": "b"}]

    async def test_calls_without_ids_get_their_own(self, driver: Driver, mode: str) -> None:
        if mode == "generate":
            driver.require(RESPONSE_CALL_WITHOUT_ID)
        script = Script(
            calls=(
                Call("lookup", '{"q": "a"}', index=0),
                Call("lookup", '{"q": "b"}', index=1),
            ),
            finish="tool",
        )

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        ids = [c.id for c in answer.calls]
        assert len(ids) == 2 and all(ids) and len(set(ids)) == 2

    async def test_calls_sharing_a_server_id_get_their_own(
        self, driver: Driver, mode: str
    ) -> None:
        driver.require(REPEATED_ID)
        script = Script(
            calls=(
                Call("lookup", '{"q": "a"}', id="dup", index=0),
                Call("lookup", '{"q": "b"}', id="dup", index=1),
            ),
            finish="tool",
        )

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert [c.arguments for c in answer.calls] == [{"q": "a"}, {"q": "b"}]
        assert len({c.id for c in answer.calls}) == 2

    async def test_calls_without_an_index_stay_apart(self, driver: Driver) -> None:
        driver.require(CALL_INDEX)
        script = Script(
            calls=(Call("lookup", '{"q": "a"}', id="c1"), Call("lookup", '{"q": "b"}', id="c2")),
            finish="tool",
        )

        answer = await generation(driver, script, "stream", tool_context(LOOKUP))

        assert [(c.id, c.arguments) for c in answer.calls] == [
            ("c1", {"q": "a"}),
            ("c2", {"q": "b"}),
        ]

    @pytest.mark.parametrize("ids", [("c1", "c2"), ("dup", "dup")], ids=["apart", "shared"])
    async def test_composition_names_each_call_by_the_id_it_ends_with(
        self, driver: Driver, ids: tuple[str, str]
    ) -> None:
        driver.require(COMPOSITION)
        if ids[0] == ids[1]:
            driver.require(REPEATED_ID)
        script = Script(
            calls=(
                Call("lookup", '{"q": "a"}', id=ids[0], index=0, fragments=3),
                Call("lookup", '{"q": "b"}', id=ids[1], index=1, fragments=3),
            ),
            finish="tool",
        )

        answer = await generation(driver, script, "stream", tool_context(LOOKUP))

        assert answer.deltas
        assert {d.id for d in answer.deltas} <= {c.id for c in answer.calls}


class TestCallsThatDoNotRun:
    async def test_a_call_the_output_cap_cut_is_partial(self, driver: Driver, mode: str) -> None:
        driver.require(ARGUMENT_TEXT)
        script = Script(calls=(Call("lookup", '{"q": "par', id="c1", index=0),), finish="cut")

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        [call] = answer.calls
        assert (call.partial, call.garbled) == (True, False)

    async def test_a_stream_that_stops_without_a_reason_cuts_its_call(
        self, driver: Driver
    ) -> None:
        driver.require(ARGUMENT_TEXT, STREAM_WITHOUT_FINISH)
        script = Script(calls=(Call("lookup", '{"q": "par', id="c1", index=0),), finish="none")

        answer = await generation(driver, script, "stream", tool_context(LOOKUP))

        [call] = answer.calls
        assert (call.partial, call.garbled) == (True, False)

    async def test_arguments_written_unreadable_are_garbled(
        self, driver: Driver, mode: str
    ) -> None:
        driver.require(ARGUMENT_TEXT, WRITTEN_UNREADABLE)
        script = Script(calls=(Call("lookup", "[1, 2]", id="c1", index=0),), finish="tool")

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        [call] = answer.calls
        assert (call.partial, call.garbled) == (True, True)


class TestReasoningReceived:
    async def test_a_signed_block_keeps_its_signature(self, driver: Driver, mode: str) -> None:
        driver.require(SIGNED_REASONING)
        script = Script(
            reasoning=(Reasoning("why", signature="S0"),),
            calls=(Call("lookup", '{"q": "a"}', id="c1", index=0),),
            finish="tool",
        )

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert [(p.thinking, p.signature) for p in answer.reasoning] == [("why", "S0")]

    async def test_a_redacted_block_keeps_its_data(self, driver: Driver, mode: str) -> None:
        driver.require(REDACTED_REASONING)
        script = Script(
            reasoning=(Reasoning(redacted="RRR"),),
            calls=(Call("lookup", '{"q": "a"}', id="c1", index=0),),
            finish="tool",
        )

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert [p.redacted for p in answer.reasoning] == ["RRR"]


class TestUsage:
    async def test_usage_reaches_the_loop(self, driver: Driver, mode: str) -> None:
        if mode == "stream":
            driver.require(STREAM_USAGE)
        # Cache reads and reasoning only where the wire reports them apart: a
        # wire without the breakdown counts them inside input and output.
        cache = 0 if CACHE_USAGE in driver.cannot else 5
        reasoning = 0 if REASONING_USAGE in driver.cannot else 3
        script = Script(
            text="ok", usage=Usage(input=11, output=7, cache_read=cache, reasoning=reasoning)
        )

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert (answer.usage["input_tokens"], answer.usage["output_tokens"]) == (11, 7)
        if cache:
            assert answer.usage.get("cache_read_input_tokens") == cache
        if reasoning:
            assert answer.usage.get("reasoning_tokens") == reasoning
