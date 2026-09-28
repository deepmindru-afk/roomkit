"""An AI provider config carries no temperature: the turn's ``AIContext`` does (RMK-243).

The field existed on nine configs and was never read, since ``AIChannel`` always
sets ``AIContext.temperature``; a caller setting it on the config changed
nothing and was not told. Passing it now is ignored, as it was before.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.azure.config import AzureAIConfig
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.llamacpp.config import LlamaCppConfig
from roomkit.providers.mistral.config import MistralConfig
from roomkit.providers.ollama.config import OllamaConfig
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.polargrid.config import PolarGridConfig
from roomkit.providers.vllm.config import VLLMConfig

CONFIGS: list[type[BaseModel]] = [
    AnthropicConfig,
    AzureAIConfig,
    GeminiConfig,
    LlamaCppConfig,
    MistralConfig,
    OllamaConfig,
    OpenAIConfig,
    PolarGridConfig,
    VLLMConfig,
]


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.__name__)
def test_the_config_has_no_temperature_field(config: type[BaseModel]) -> None:
    assert "temperature" not in config.model_fields


def test_passing_one_is_ignored_as_before() -> None:
    config = OpenAIConfig(api_key="k", model="m", temperature=0.2)  # type: ignore[call-arg]

    assert not hasattr(config, "temperature")
