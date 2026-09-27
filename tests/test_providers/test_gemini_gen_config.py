"""The generation config the Gemini AI provider sends, built with the real SDK types."""

from __future__ import annotations

import pytest

genai_types = pytest.importorskip("google.genai.types")

from roomkit.providers.ai.base import AIContext, AIMessage  # noqa: E402
from roomkit.providers.gemini.config import GeminiConfig  # noqa: E402
from roomkit.providers.gemini.request import build_gen_config  # noqa: E402


def _gen_config(*, thinking_level: str | None = None, thinking_budget: int | None = None):
    config = GeminiConfig(api_key="test-key", thinking_level=thinking_level)
    context = AIContext(
        messages=[AIMessage(role="user", content="Hi")], thinking_budget=thinking_budget
    )
    return build_gen_config(genai_types, config, context)


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

    def test_the_configured_level_still_wins_over_a_turn_budget(self) -> None:
        gen = _gen_config(thinking_level="low", thinking_budget=0)

        assert gen.thinking_config.thinking_level == genai_types.ThinkingLevel.LOW
        assert gen.thinking_config.thinking_budget is None
