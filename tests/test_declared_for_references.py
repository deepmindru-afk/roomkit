"""A provider that holds no tool unseen gets each tool result as its text,
never the references only a deferring provider reads (RMK-484, RFC §6.4).

Anthropic behind a gateway does not defer tools, as OpenAI does not: the
turn's held tools it referenced are declared plainly, and the result that
referenced them goes as the text it carries, on both. Anthropic on its own
endpoint keeps the references, which expand the held definitions.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.channels._ai_policy import declared_for
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
)
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.anthropic.request import build_kwargs
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig

FIND = AITool(name="find_tools", description="d")
HELD = AITool(name="send_mail", description="d", defer_loading=True)
GATEWAY = "https://gateway.example.test/anthropic"


def _turn() -> AIContext:
    call = AIToolCallPart(id="c1", name="find_tools", arguments={"q": "mail"})
    result = AIToolResultPart(
        tool_call_id="c1",
        name="find_tools",
        result="send_mail: send a mail",
        references=["send_mail"],
    )
    history = [
        AIMessage(role="user", content="mail bob"),
        AIMessage(role="assistant", content=[call]),
        AIMessage(role="tool", content=[result]),
    ]
    return AIContext(messages=history, tools=[FIND, HELD])


def _anthropic(base_url: str | None = None) -> AnthropicConfig:
    return AnthropicConfig(api_key="k", model="claude-sonnet-5-5", base_url=base_url)


NON_DEFERRING: dict[str, Any] = {
    "anthropic-gateway": lambda: AnthropicAIProvider(_anthropic(GATEWAY)),
    "openai": lambda: OpenAIAIProvider(OpenAIConfig(api_key="k", model="gpt-4.1")),
}


def _results(context: AIContext) -> list[AIToolResultPart]:
    return [
        part
        for message in context.messages
        if isinstance(message.content, list)
        for part in message.content
        if isinstance(part, AIToolResultPart)
    ]


@pytest.mark.parametrize("provider", list(NON_DEFERRING))
def test_a_non_deferring_provider_gets_the_result_text_and_the_tool_declared(
    provider: str,
) -> None:
    context = declared_for(NON_DEFERRING[provider](), _turn())

    assert [(p.result, p.references) for p in _results(context)] == [
        ("send_mail: send a mail", [])
    ]
    assert [(t.name, t.defer_loading) for t in context.tools] == [
        ("find_tools", False),
        ("send_mail", False),
    ]


def test_anthropic_behind_a_gateway_sends_the_result_as_text() -> None:
    config = _anthropic(GATEWAY)
    context = declared_for(AnthropicAIProvider(config), _turn())

    kwargs = build_kwargs(config, context)

    assert kwargs["messages"][2]["content"][0]["content"] == "send_mail: send a mail"


def test_anthropic_on_its_own_endpoint_keeps_the_references() -> None:
    config = _anthropic()
    context = declared_for(AnthropicAIProvider(config), _turn())

    kwargs = build_kwargs(config, context)

    assert kwargs["messages"][2]["content"][0]["content"] == [
        {"type": "tool_reference", "tool_name": "send_mail"}
    ]
