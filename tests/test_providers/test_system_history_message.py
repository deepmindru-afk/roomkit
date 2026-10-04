"""A ``system`` message in the history goes as a user turn on the wires
whose messages take no system role (RMK-484, RFC §6.7).

A memory provider's summary or an instruction rides the history with the role
``system``; the Messages API and Gemini take user and assistant (model) turns
only, the system prompt going apart. Anthropic folds it into a user turn as
Gemini does, instead of sending a role the API refuses.
"""

from __future__ import annotations

from google.genai import types

from roomkit.providers.ai.base import AIContext, AIMessage
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.anthropic.request import build_kwargs
from roomkit.providers.gemini.request import format_messages

HISTORY = [
    AIMessage(role="system", content="Summary of the conversation"),
    AIMessage(role="user", content="go"),
]


def test_anthropic_sends_a_system_message_as_a_user_turn() -> None:
    config = AnthropicConfig(api_key="k", model="claude-sonnet-5-5")

    kwargs = build_kwargs(config, AIContext(messages=HISTORY, system_prompt="Be brief."))

    assert [m["role"] for m in kwargs["messages"]] == ["user", "user"]
    assert "Summary of the conversation" in str(kwargs["messages"][0]["content"])
    assert "Be brief." in str(kwargs["system"])


def test_gemini_sends_it_as_a_user_turn_too() -> None:
    assert [content.role for content in format_messages(types, HISTORY)] == ["user", "user"]
