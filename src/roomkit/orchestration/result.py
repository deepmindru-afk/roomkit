"""Structured result handoff for orchestration.

A delegated agent returns its work by calling the ``submit_result`` tool rather
than ending its turn with a free-text message that gets scraped from the room.
This forces a structured, parseable handoff — and a result at all (a worker
can't punt with a question) — so the next step and the supervisor receive a
clean object instead of prose. Reusable by any orchestration that delegates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from roomkit.providers.ai.base import AITool

SUBMIT_RESULT_TOOL_NAME = "submit_result"


def is_submit_result(tool_name: str) -> bool:
    """Whether *tool_name* is a submit_result call.

    A function-calling provider calls the injected tool by its bare name; a
    claude_code worker calls the gateway-exposed tool, which the sandbox surfaces
    with an MCP prefix (``mcp__<server>__submit_result``). Match both so
    the capture is delivery-agnostic.
    """
    return SUBMIT_RESULT.matches(tool_name)


#: Injected into a delegated worker so it delivers its result through a tool
#: call (forced structure) instead of a scraped free-text message.
SUBMIT_RESULT_TOOL = AITool(
    name=SUBMIT_RESULT_TOOL_NAME,
    description=(
        "Submit your FINAL result for this task. You MUST call this exactly once, "
        "when your work is done — it is the ONLY way to hand your work to the next "
        "step. Do NOT end your turn with a plain message or a question back to the "
        "user; call submit_result with your structured result."
    ),
    parameters={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["completed", "failed"],
                "description": (
                    "completed if you did the task; failed only if you genuinely could not."
                ),
            },
            "summary": {
                "type": "string",
                "description": "One or two sentences summarizing what you produced.",
            },
            "data": {
                "type": "object",
                "description": "Your structured result, for the next step to build on.",
            },
            "deliverables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                    },
                },
                "description": "Concrete artifacts you produced (e.g. a published report URL).",
            },
            "reason": {
                "type": "string",
                "description": "If status is failed, explain why.",
            },
        },
        "required": ["status", "summary"],
    },
)


def normalize_result(arguments: dict[str, Any]) -> dict[str, Any]:
    """Coerce a ``submit_result`` tool-call payload into the canonical shape."""
    data = arguments.get("data")
    deliverables = arguments.get("deliverables")
    return {
        "status": arguments.get("status") or "completed",
        "summary": str(arguments.get("summary") or ""),
        "data": data if isinstance(data, dict) else {},
        "deliverables": deliverables if isinstance(deliverables, list) else [],
        "reason": str(arguments.get("reason") or ""),
    }


def orchestration_fail(*, role: str, last_output: str, attempts: int) -> dict[str, Any]:
    """Fail payload the orchestration submits on a worker's behalf when it never
    called ``submit_result`` after *attempts* tries.

    Distinct from a worker self-reporting a task failure (``status="failed"`` via
    the tool): ``by="orchestration"`` marks a MECHANISM-level failure, carrying
    the worker's last raw output as explanatory context so the next step and the
    supervisor understand precisely what went wrong.
    """
    return {
        "status": "failed",
        "by": "orchestration",
        "reason": f"no_structured_result_after_{attempts}_attempts",
        "role": role,
        "last_output": last_output,
    }


@dataclass(frozen=True)
class ResultTool:
    """The tool a delegated agent must call to hand its work back, and how it is read.

    A delegation run with ``require_structured_result`` injects :attr:`tool`,
    captures its call, reminds the agent with :attr:`reminder` when a turn ends
    without it, and gives up after the retries with :attr:`on_missing`. The call
    is how the structure is forced, so it works with any provider that calls
    tools, whatever its support for a response schema. :data:`SUBMIT_RESULT` (a
    worker's result) is the default; the supervised flow hands its verdict back
    through its own tool.

    Attributes:
        tool: The tool injected into the agent for the delegation.
        normalize: Turns the call's arguments into the payload returned.
        on_missing: Builds the payload returned when the agent never called the
            tool, from ``role``, ``last_output`` and ``attempts`` (keywords).
        reminder: The message sent when a turn ends without the call.
    """

    tool: AITool
    normalize: Callable[[dict[str, Any]], dict[str, Any]]
    on_missing: Callable[..., dict[str, Any]]
    reminder: str

    @property
    def name(self) -> str:
        """The tool's name."""
        return self.tool.name

    def matches(self, tool_name: str) -> bool:
        """Whether *tool_name* is a call of this tool.

        A function-calling provider calls it by its bare name; a claude_code
        worker calls the gateway-exposed tool, which the sandbox surfaces with an
        MCP prefix (``mcp__<server>__<name>``). Both count.
        """
        return tool_name == self.name or tool_name.endswith(f"__{self.name}")


#: A worker's structured result: the default a delegation forces.
SUBMIT_RESULT = ResultTool(
    tool=SUBMIT_RESULT_TOOL,
    normalize=normalize_result,
    on_missing=orchestration_fail,
    reminder=(
        "You did not submit a result. You MUST now call the `submit_result` "
        "tool with your final structured result. Do NOT reply with plain text "
        "or a question — call submit_result."
    ),
)
