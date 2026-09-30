"""The generation config the Gemini AI provider sends, built with the real SDK types."""

from __future__ import annotations

from typing import Any

import pytest

genai_types = pytest.importorskip("google.genai.types")

from roomkit.providers.ai.base import AIContext, AIMessage  # noqa: E402
from roomkit.providers.gemini.config import GeminiConfig  # noqa: E402
from roomkit.providers.gemini.request import build_gen_config  # noqa: E402


def _gen_config(
    *, thinking_level: str | None = None, capabilities: tuple[str, ...] = (), **turn: Any
):
    config = GeminiConfig(api_key="test-key", thinking_level=thinking_level)
    context = AIContext(messages=[AIMessage(role="user", content="Hi")], **turn)
    return build_gen_config(genai_types, config, context, capabilities)


class TestAutomaticFunctionCalling:
    def test_it_is_off(self) -> None:
        """RoomKit runs its own tool loop; left on, the SDK logs a warning per call."""
        gen = _gen_config()

        assert gen.automatic_function_calling is not None
        assert gen.automatic_function_calling.disable is True


class TestThinkingBudget:
    def test_zero_turns_reasoning_off_as_on_every_other_provider(self) -> None:
        """``0`` used to read as "not set", so reasoning could not be turned off."""
        gen = _gen_config(thinking_budget=0)

        assert gen.thinking_config is not None
        assert gen.thinking_config.thinking_budget == 0
        # Off means no summaries to ask for.
        assert not gen.thinking_config.include_thoughts

    def test_a_budget_is_sent_with_its_summaries(self) -> None:
        gen = _gen_config(thinking_budget=512)

        assert gen.thinking_config.thinking_budget == 512
        assert gen.thinking_config.include_thoughts is True

    def test_dynamic_budget_keeps_its_summaries(self) -> None:
        gen = _gen_config(thinking_budget=-1)

        assert gen.thinking_config.thinking_budget == -1
        assert gen.thinking_config.include_thoughts is True

    def test_unset_sends_no_thinking_config(self) -> None:
        assert _gen_config().thinking_config is None

    def test_a_turn_budget_of_zero_turns_off_a_configured_level(self) -> None:
        """RFC §6.7: the turn outranks the vendor setting on what it states."""
        gen = _gen_config(thinking_level="high", thinking_budget=0)

        assert gen.thinking_config.thinking_level is None
        assert gen.thinking_config.thinking_budget == 0

    def test_a_turn_budget_that_turns_thinking_on_keeps_the_configured_level(self) -> None:
        gen = _gen_config(thinking_level="high", thinking_budget=4096)

        assert gen.thinking_config.thinking_level == genai_types.ThinkingLevel.HIGH
        assert gen.thinking_config.thinking_budget is None


_LEVELS = ("thinking_level",)
_MINIMAL = ("thinking_level", "thinking_level_minimal")


class TestTheTurnsLevel:
    """RFC §6.7: ``reasoning_effort`` states the level, where the model takes one."""

    @pytest.mark.parametrize(
        ("effort", "capabilities", "level"),
        [
            ("low", _LEVELS, "LOW"),
            ("xhigh", _LEVELS, "HIGH"),
            ("minimal", _LEVELS, "LOW"),
            ("minimal", _MINIMAL, "MINIMAL"),
        ],
    )
    def test_the_effort_is_sent_as_the_nearest_level_the_model_takes(
        self, effort: str, capabilities: tuple[str, ...], level: str
    ) -> None:
        gen = _gen_config(reasoning_effort=effort, capabilities=capabilities)

        assert gen.thinking_config.thinking_level == genai_types.ThinkingLevel[level]

    def test_the_effort_replaces_a_configured_level(self) -> None:
        gen = _gen_config(thinking_level="high", reasoning_effort="low")

        assert gen.thinking_config.thinking_level == genai_types.ThinkingLevel.LOW

    def test_a_model_without_levels_is_sent_none(self) -> None:
        """Gemini 2.5 refuses a level (400); the catalogue gives it no tag."""
        assert _gen_config(reasoning_effort="low").thinking_config is None

    def test_an_effort_of_none_turns_thinking_off(self) -> None:
        gen = _gen_config(thinking_level="high", reasoning_effort="none", capabilities=_LEVELS)

        assert gen.thinking_config.thinking_budget == 0

    def test_enable_thinking_alone_turns_on_a_dynamic_budget(self) -> None:
        gen = _gen_config(enable_thinking=True)

        assert gen.thinking_config.thinking_budget == -1
        assert gen.thinking_config.include_thoughts is True


class TestResponseSchema:
    """RFC §6.7: controlled generation, with the real SDK types."""

    def test_the_schema_sets_json_mime_type_and_response_json_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": ["yes", "no"]}},
            "required": ["label"],
            "additionalProperties": False,
        }
        config = GeminiConfig(api_key="test-key")
        context = AIContext(
            messages=[AIMessage(role="user", content="Hi")], response_schema=schema
        )

        gen = build_gen_config(genai_types, config, context)

        assert gen.response_mime_type == "application/json"
        assert gen.response_json_schema == schema

    def test_no_schema_leaves_both_unset(self) -> None:
        gen = _gen_config()

        assert gen.response_mime_type is None
        assert gen.response_json_schema is None
