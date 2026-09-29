"""Tool declarations the Gemini SDK builds, or refuses by name (RMK-281)."""

from __future__ import annotations

from typing import Any

import pytest

types = pytest.importorskip("google.genai", reason="google-genai not installed").types

from roomkit.providers.ai.base import AIContext, AIMessage, AITool, ProviderError  # noqa: E402
from roomkit.providers.gemini.ai import GeminiAIProvider  # noqa: E402
from roomkit.providers.gemini.config import GeminiConfig  # noqa: E402
from roomkit.providers.gemini.realtime_config import build_live_config  # noqa: E402
from roomkit.providers.gemini.schema import function_declaration  # noqa: E402


@pytest.mark.parametrize(
    "prop",
    [
        {"type": "integer", "enum": [1, 2, 3]},
        {"type": "number", "enum": [0.5, 1.5]},
        {"type": "boolean", "enum": [True]},
        {"enum": ["a", 1, None]},
        {"type": "array", "items": {"enum": [1, 2]}},
        {"type": "array", "items": [{"type": "number"}, {"type": "number"}]},
    ],
)
def test_a_schema_the_sdk_refused_is_declarable(prop: dict[str, Any]) -> None:
    schema = {"type": "object", "properties": {"p": prop}}

    declaration = function_declaration(types, name="t", description="d", parameters=schema)

    assert declaration.name == "t"


# The cleaner lets a wrongly typed keyword through; the SDK refuses it.
_UNDECLARABLE = {"type": "object", "properties": {"rate": {"type": "integer", "minimum": "low"}}}


class TestADeclarationGeminiRefuses:
    def test_the_helper_names_the_tool(self) -> None:
        with pytest.raises(ProviderError, match="'set_rate'") as raised:
            function_declaration(types, name="set_rate", description="d", parameters=_UNDECLARABLE)

        assert raised.value.provider == "gemini"
        assert raised.value.retryable is False
        assert "minimum" in str(raised.value)

    async def test_a_text_turn_raises_it(self) -> None:
        provider = GeminiAIProvider(GeminiConfig(api_key="test-api-key"))
        context = AIContext(
            messages=[AIMessage(role="user", content="Hi")],
            tools=[AITool(name="set_rate", description="d", parameters=_UNDECLARABLE)],
        )

        with pytest.raises(ProviderError, match="'set_rate'"):
            await provider.generate(context)

    def test_a_live_config_raises_it(self) -> None:
        tool = {"name": "set_rate", "description": "d", "parameters": _UNDECLARABLE}

        with pytest.raises(ProviderError, match="'set_rate'"):
            build_live_config("gemini-3.8-live", tools=[tool])
