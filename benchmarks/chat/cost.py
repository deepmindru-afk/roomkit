"""A fixed tool conversation, billed as a real provider bills it.

The model's answers are scripted, so every run makes the same requests over
the same rounds, in the streaming loop and the buffered one. Each request the
channel builds is also sent to the real provider, whose usage (input, cache
read, cache write) is what the report counts; its answer is discarded. That
is what lets a change to how a request is assembled be measured before and
after on identical traffic. The conversation itself is in ``cost_script``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from benchmarks.chat.cost_script import (
    FORCE_STOP_TURNS,
    LONG_TURNS,
    SYSTEM,
    TOOL_TURNS,
    Turns,
    catalogue,
    serve,
)
from benchmarks.chat.harness import Harness
from benchmarks.chat.scenarios import Scenario, record_handler
from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    AITool,
    ModelInfo,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills import SkillRegistry


class BilledScript(MockAIProvider):
    """Answer each round from the script, billed as *billing* bills the same request.

    The request goes to *billing* with ``max_tokens=1``: its answer is
    discarded, its usage is stamped on the scripted response. With no
    *billing* (an offline run) the usage stays empty and only the requests
    are observed.
    """

    def __init__(self, billing: AIProvider | None, responses: list[AIResponse]) -> None:
        super().__init__(ai_responses=responses, streaming=True)
        self._billing = billing
        self._scripted = len(responses)

    @property
    def model_name(self) -> str:
        return self._billing.model_name if self._billing is not None else "mock"

    @property
    def context_window(self) -> int | None:
        return self._billing.context_window if self._billing is not None else None

    def catalog_entry(self) -> ModelInfo | None:
        return self._billing.catalog_entry() if self._billing is not None else None

    @property
    def supports_deferred_tools(self) -> bool:
        return self._billing is not None and self._billing.supports_deferred_tools

    @property
    def consumed(self) -> bool:
        """Whether the turns asked for exactly the scripted rounds."""
        return self._index == self._scripted

    async def generate(self, context: AIContext) -> AIResponse:
        response = await super().generate(context)
        if self._billing is None:
            return response
        billed = await self._billing.generate(context.model_copy(update={"max_tokens": 1}))
        return response.model_copy(update={"usage": dict(billed.usage)})


def first_change(previous: AIContext | None, current: AIContext) -> str:
    """Name the first block of *current* that differs from *previous*.

    Blocks are read in the order a prompt cache reads a request: the tools,
    the system prompt, then the messages. ``append`` means the whole previous
    request is a prefix of this one, which a cache can read back; any other
    name is where a cached prefix stops.
    """
    if previous is None:
        return "first"
    if _tools_key(previous.tools) != _tools_key(current.tools):
        return "tools"
    if previous.system_prompt != current.system_prompt:
        return "system"
    if current.messages[: len(previous.messages)] != previous.messages:
        return "messages"
    return "append"


def _tools_key(tools: list[AITool]) -> list[tuple[str, str, str]]:
    # Unsorted: a provider's cache is byte-exact, so reordered keys are a change.
    return [(t.name, t.description, json.dumps(t.parameters)) for t in tools]


def cost_rows(h: Harness, turn_starts: list[int]) -> list[dict[str, Any]]:
    """One row per provider call: its turn, round, first changed block,
    usage and cost."""
    entry = h.provider.catalog_entry()
    pricing = entry.pricing if entry is not None else None
    rows: list[dict[str, Any]] = []
    previous: AIContext | None = None
    for index, (call, context) in enumerate(
        zip(h.provider.calls, h.provider.contexts, strict=True)
    ):
        turn = sum(1 for start in turn_starts if start <= index)
        usage = {k: v for k, v in call.usage.items() if isinstance(v, int)}
        rows.append(
            {
                "turn": turn,
                "round": index - turn_starts[turn - 1],
                "change": first_change(previous, context),
                "tools": len(context.tools),
                "input_tokens": usage.get("input_tokens", 0),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cost": pricing.cost_for(usage) if pricing is not None else None,
            }
        )
        previous = context
    return rows


def _conversation(turns: Turns) -> Callable[[Harness], Awaitable[None]]:
    """The scenario that says each turn and records every round's cost."""

    async def run(h: Harness) -> None:
        h.handler = record_handler(h, serve)
        turn_starts: list[int] = []
        for text, _ in turns:
            turn_starts.append(len(h.provider.calls))
            result = await h.ask(text)
            h.check("turn_answered", not result.blocked and not result.error)
        script = h.provider.inner
        h.check("script_consumed", isinstance(script, BilledScript) and script.consumed)
        h.details["cost"] = cost_rows(h, turn_starts)

    return run


def _billed(turns: Turns) -> Callable[[AIProvider | None], AIProvider]:
    """The model that plays *turns*, billed by the provider it is handed."""
    responses = [response for _, rounds in turns for response in rounds]

    def model(billing: AIProvider | None) -> AIProvider:
        return BilledScript(billing, responses)

    return model


def cost_options(nonce: str | None = None) -> dict[str, Any]:
    """The channel of every cost scenario: the catalogue behind Tool Search,
    two pinned tools, the quote-policy skill, all marked with *nonce*."""
    registry = SkillRegistry()
    registry.discover(Path(__file__).parent / "fixtures")
    # One run must not read another's cache: the nonce opens the tools and the
    # system prompt, since a request may carry either without the other.
    nonce = nonce or uuid.uuid4().hex[:12]
    return {
        "system_prompt": f"[bench run {nonce}] {SYSTEM}",
        "tools": catalogue(nonce),
        "tool_search": True,
        "tool_search_pinned": {"lookup_order", "inventory"},
        "skills": registry,
    }


# Each conversation of the suite: its name, what it exercises, its turns.
CONVERSATIONS: list[tuple[str, str, Turns]] = [
    ("tool_cost", "Tool Search reveal, eviction, skill, digest over four turns", TOOL_TURNS),
    ("force_stop_cost", "Six identical calls, then the anti-loop ripcord", FORCE_STOP_TURNS),
    ("long_cost", "Six long turns of history, then five turns with tools", LONG_TURNS),
]


def cost_scenarios() -> list[Scenario]:
    """Each conversation, in the streaming loop and in the buffered one."""
    return [
        Scenario(
            name if streaming else name + "_buffered",
            ("cost", "tools", "cache"),
            description + ("" if streaming else " (buffered)"),
            _conversation(turns),
            streaming=streaming,
            options_factory=cost_options,
            mock_supported=True,
            model=_billed(turns),
        )
        for name, description, turns in CONVERSATIONS
        for streaming in (True, False)
    ]
