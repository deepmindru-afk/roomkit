"""Response schema — an AI provider answers in a JSON shape you declare.

A support message is triaged into a department, an urgency flag and a one-line
summary. ``AIContext.response_schema`` carries the shape; the provider
constrains its output natively (OpenAI ``response_format``, Anthropic
``output_config``, Gemini ``response_json_schema``, Mistral, Ollama, PolarGrid)
and ``generate()`` returns one JSON document, or raises ``ResponseSchemaError``
saying why it could not: a refusal, a truncated answer, text that is not JSON,
or a provider that cannot take a schema at all.

The schema stays within the portable subset (RFC §6.7): every object lists all
its properties in ``required`` and sets ``additionalProperties`` to false, and
only strings carry ``enum``. That subset runs on every provider that supports
response schemas.

Run with:
    uv run python examples/ai_response_schema.py                     # offline, mock
    GEMINI_API_KEY=... uv run python examples/ai_response_schema.py gemini
    ANTHROPIC_API_KEY=... uv run python examples/ai_response_schema.py anthropic
    OPENAI_API_KEY=... uv run python examples/ai_response_schema.py openai
    uv run python examples/ai_response_schema.py ollama              # local server
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import asyncio
import json

from shared import require_env

from roomkit import ResponseSchemaError
from roomkit.providers.ai import AIContext, AIMessage, AIProvider, MockAIProvider
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.providers.gemini import GeminiAIProvider, GeminiConfig
from roomkit.providers.ollama import OllamaAIProvider, OllamaConfig
from roomkit.providers.openai import OpenAIAIProvider, OpenAIConfig

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "department": {
            "type": "string",
            "enum": ["billing", "technical", "sales", "other"],
            "description": "The team that should take the message.",
        },
        "urgent": {"type": "boolean", "description": "Whether the customer threatens to leave."},
        "summary": {"type": "string", "description": "One short sentence."},
    },
    "required": ["department", "urgent", "summary"],
    "additionalProperties": False,
}

MESSAGE = (
    "I was charged twice for March. This is the third time I'm calling. "
    "If this isn't fixed today I'm cancelling."
)


def build_provider(name: str) -> AIProvider:
    """The provider named on the command line; the mock needs no key."""
    if name == "gemini":
        env = require_env("GEMINI_API_KEY")
        return GeminiAIProvider(GeminiConfig(api_key=env["GEMINI_API_KEY"]))
    if name == "anthropic":
        env = require_env("ANTHROPIC_API_KEY")
        return AnthropicAIProvider(
            AnthropicConfig(api_key=env["ANTHROPIC_API_KEY"], model="claude-opus-5")
        )
    if name == "openai":
        env = require_env("OPENAI_API_KEY")
        return OpenAIAIProvider(OpenAIConfig(api_key=env["OPENAI_API_KEY"], model="gpt-5.6-sol"))
    if name == "ollama":
        return OllamaAIProvider(OllamaConfig())
    answer = {
        "department": "billing",
        "urgent": True,
        "summary": "Double charge for March, third call, threatens to cancel.",
    }
    return MockAIProvider([json.dumps(answer)], response_schema=True)


async def main(provider_name: str) -> None:
    provider = build_provider(provider_name)
    if not provider.supports_response_schema:
        # A consumer meant for any provider falls back to asking for JSON in
        # the prompt and parsing it itself; this example just says so.
        print(f"{provider.name} does not constrain its output to a schema.")
        return

    context = AIContext(
        system_prompt="You triage customer support messages.",
        messages=[AIMessage(role="user", content=MESSAGE)],
        response_schema=TRIAGE_SCHEMA,
        max_tokens=1024,
    )
    try:
        response = await provider.generate(context)
    except ResponseSchemaError as exc:
        print(f"No triage ({exc.reason}): {exc}")
        return
    finally:
        await provider.close()

    triage = json.loads(response.content)
    print(f"Provider:   {provider.name} ({provider.model_name})")
    print(f"Department: {triage['department']}")
    print(f"Urgent:     {triage['urgent']}")
    print(f"Summary:    {triage['summary']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "provider",
        nargs="?",
        default="mock",
        choices=["mock", "gemini", "anthropic", "openai", "ollama"],
    )
    asyncio.run(main(parser.parse_args().provider))
