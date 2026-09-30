"""The turn's reasoning settings outrank the provider's configuration (RFC §6.7).

One table of every provider whose configuration carries a reasoning setting
under the name the turn uses, and of where each puts it on the request: a
provider that reads its configuration alone fails here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, AITool
from roomkit.providers.ai.reasoning import turn_setting
from roomkit.providers.cerebras.ai import CerebrasAIProvider
from roomkit.providers.cerebras.config import CerebrasConfig
from roomkit.providers.deepseek.ai import DeepSeekAIProvider
from roomkit.providers.deepseek.config import DeepSeekConfig
from roomkit.providers.litellm.ai import LiteLLMAIProvider
from roomkit.providers.litellm.config import LiteLLMConfig
from roomkit.providers.meta.ai import MetaAIProvider
from roomkit.providers.meta.config import MetaConfig
from roomkit.providers.mistral.ai import MistralAIProvider
from roomkit.providers.mistral.config import MistralConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.openrouter.ai import OpenRouterAIProvider
from roomkit.providers.openrouter.config import OpenRouterConfig
from roomkit.providers.qwen.ai import QwenAIProvider
from roomkit.providers.qwen.config import QwenConfig
from roomkit.providers.xai.ai import XAIAIProvider
from roomkit.providers.xai.config import XAIConfig

_TOOLS = [AITool(name="lookup", description="x", parameters={})]


def _provider(cls: type, config: Any) -> Any:
    provider = cls.__new__(cls)
    provider._config = config
    return provider


def _sampled(provider: Any, context: AIContext) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    provider._apply_sampling_kwargs(kwargs, context)
    return kwargs


def _top_level(provider: Any, context: AIContext) -> Any:
    return _sampled(provider, context).get("reasoning_effort")


def _openrouter(provider: Any, context: AIContext) -> Any:
    return _sampled(provider, context).get("extra_body", {}).get("reasoning", {}).get("effort")


def _deepseek(provider: Any, context: AIContext) -> Any:
    thinking = _sampled(provider, context).get("extra_body", {}).get("thinking", {})
    return thinking.get("reasoning_effort")


def _mistral(provider: Any, context: AIContext) -> Any:
    return provider._resolve_reasoning_effort(context)


Effort = tuple[Callable[[str], Any], Callable[[Any, AIContext], Any]]


def _with(cls: type, config: type, **fields: Any) -> Callable[[str], Any]:
    """Build *cls* with a configured effort."""
    return lambda effort: _provider(cls, config(api_key="k", reasoning_effort=effort, **fields))


# Each provider built with a configured effort, and how to read the effort it sends.
_EFFORT: dict[str, Effort] = {
    "openai": (_with(OpenAIAIProvider, OpenAIConfig, model="gpt-5-mini"), _top_level),
    "openrouter": (_with(OpenRouterAIProvider, OpenRouterConfig, model="m"), _openrouter),
    "xai": (_with(XAIAIProvider, XAIConfig), _top_level),
    "cerebras": (_with(CerebrasAIProvider, CerebrasConfig, model="m"), _top_level),
    "meta": (_with(MetaAIProvider, MetaConfig), _top_level),
    "litellm": (_with(LiteLLMAIProvider, LiteLLMConfig, model="m"), _top_level),
    "deepseek": (_with(DeepSeekAIProvider, DeepSeekConfig, model="m"), _deepseek),
    "mistral": (_with(MistralAIProvider, MistralConfig), _mistral),
}

# Where a turn with tools does not carry an enabling effort, and why (RFC §6.7):
# LiteLLM cannot know the model behind an alias; OpenRouter cannot pass a
# tool round's reasoning back as its upstreams require.
_NOT_ON_TOOL_TURNS = {"litellm", "openrouter"}


def _context(**fields: Any) -> AIContext:
    return AIContext(messages=[AIMessage(role="user", content="hi")], **fields)


def test_a_turn_value_outranks_the_configured_one_even_when_falsy() -> None:
    assert turn_setting(False, True) is False
    assert turn_setting(None, True) is True
    assert turn_setting("low", "high") == "low"
    assert turn_setting(None, None) is None


@pytest.mark.parametrize("name", sorted(_EFFORT))
def test_the_turn_effort_outranks_the_configured_one(name: str) -> None:
    build, read = _EFFORT[name]

    assert read(build("high"), _context(reasoning_effort="low")) == "low"
    assert read(build("high"), _context()) == "high"


@pytest.mark.parametrize("name", sorted(set(_EFFORT) - _NOT_ON_TOOL_TURNS))
def test_a_tool_turn_carries_the_effort_as_a_turn_without_does(name: str) -> None:
    build, read = _EFFORT[name]

    assert read(build("high"), _context(tools=_TOOLS, reasoning_effort="low")) == "low"


_SWITCHED_ON = {"api_key": "k", "model": "m", "enable_thinking": True}


@pytest.mark.parametrize(
    ("provider", "read"),
    [
        (
            _provider(DeepSeekAIProvider, DeepSeekConfig(**_SWITCHED_ON)),
            lambda kwargs: kwargs["extra_body"]["thinking"],
        ),
        (
            _provider(QwenAIProvider, QwenConfig(**_SWITCHED_ON)),
            lambda kwargs: kwargs["extra_body"]["enable_thinking"],
        ),
    ],
    ids=["deepseek", "qwen"],
)
def test_the_turn_switch_off_outranks_a_configured_switch_on(
    provider: Any, read: Callable[[dict[str, Any]], Any]
) -> None:
    sent = read(_sampled(provider, _context(tools=_TOOLS, enable_thinking=False)))

    assert sent in ({"type": "disabled"}, False)
