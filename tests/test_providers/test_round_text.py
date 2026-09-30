"""A tool round's reasoning reaches the next round, even when it also spoke.

The OpenAI-compatible builders pass a round's reasoning back as a leading
``<think>`` block of its content. A round that said something besides its
calls used to keep its text alone: the text overwrote the reasoning.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.providers.ai.base import AIMessage, AITextPart, AIThinkingPart, AIToolCallPart
from roomkit.providers.ai.openai_dialect import round_text
from roomkit.providers.mistral.ai import MistralAIProvider
from roomkit.providers.mistral.config import MistralConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig

_CALL = AIToolCallPart(id="c1", name="lookup", arguments={"id": "A-1"})


def _builder(cls: type, config: Any) -> Any:
    provider = cls.__new__(cls)
    provider._config = config
    return provider


@pytest.mark.parametrize(
    "provider",
    [
        _builder(OpenAIAIProvider, OpenAIConfig(api_key="k", model="gpt-4.1")),
        _builder(MistralAIProvider, MistralConfig(api_key="k")),
    ],
    ids=["openai", "mistral"],
)
def test_a_round_that_spoke_keeps_its_reasoning(provider: Any) -> None:
    round_ = AIMessage(
        role="assistant",
        content=[AIThinkingPart(thinking="A-1 first."), AITextPart(text="Let me look."), _CALL],
    )

    (sent,) = provider._build_messages([round_])

    assert sent["content"] == "<think>A-1 first.</think>Let me look."
    assert [c["id"] for c in sent["tool_calls"]] == ["c1"]


def test_a_round_is_its_reasoning_then_its_text_whatever_their_order() -> None:
    thinking, text = AIThinkingPart(thinking="why"), AITextPart(text="what")

    assert round_text([text, thinking, _CALL]) == "<think>why</think>what"
    assert round_text([thinking, _CALL]) == "<think>why</think>"
    assert round_text([text, _CALL]) == "what"
    # A signature alone (Anthropic's) carries no text to pass back.
    assert round_text([AIThinkingPart(thinking="", signature="s"), _CALL]) == ""
