"""RoomKit — tool policy and hooks in front of tools an external agent runs.

Some agents run their own tools: an ACP coding agent such as Claude Code
(``examples/acp_claude_code.py``), or a provider with its own sandbox. RoomKit
does not execute those tools, but an ``ExternalToolHandler`` still decides
whether each call may run and reports how it went:

    agent proposes a call
      -> handler.process_tool_call()   BEFORE_TOOL_USE hooks (block, rewrite),
                                        then the ToolPolicy (allow, deny, ask)
      -> ToolDecision to the agent     refused, or run with these arguments
      -> the agent runs the tool
      -> handler.on_tool_result()      ON_TOOL_CALL hooks, with is_error

The agent here is simulated in-process — no API key, no subprocess. It plays
the part ``ACPChannel`` plays for a real agent: it asks the handler before each
call and reports each result. Each line of output is printed by the part that
acted: a hook, the reviewer, or the agent reading the handler's decision.

The handler is ``PolicyExternalToolHandler`` (hooks, then allow/deny) with a
third answer added, "ask", for tools a reviewer approves call by call. The
reviewer is scripted; a real one is a terminal prompt (``acp_claude_code.py``)
or a UI. ``ACPChannel`` refuses a call whose arguments a hook rewrote, because
ACP v1 cannot hand new arguments to the agent; this simulated agent runs the
rewritten call.

Run with:
    uv run python examples/external_tool_handler.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import setup_logging

from roomkit import AIChannel, HookResult, HookTrigger, RoomContext, RoomKit, ToolCallEvent
from roomkit.providers.ai import MockAIProvider
from roomkit.tools import ExternalToolHandler, PolicyExternalToolHandler, ToolDecision, ToolPolicy

ROOM_ID = "external-tools"

# Deny: Bash never runs, whoever asks. (A non-empty ``allow`` list would also
# refuse every tool it does not name.)
POLICY = ToolPolicy(deny=["Bash"])
# Ask: allowed by the policy, but a reviewer approves each call.
ASK = ["Write"]
# Allow: everything else runs without asking.

# What the simulated agent tries, in order.
PLAN: list[tuple[str, dict[str, Any]]] = [
    ("Read", {"path": "README.md"}),
    ("Grep", {"pattern": "TODO"}),
    ("Read", {"path": ".env"}),
    ("Read", {"path": "docs/missing.md"}),
    ("Bash", {"command": "rm -rf build/"}),
    ("Write", {"path": "notes/plan.md", "content": "Fix the TODOs."}),
    ("Write", {"path": "pyproject.toml", "content": "[project]"}),
]

Reviewer = Callable[[str, dict[str, Any]], Awaitable[bool]]


def _show(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False)


class ReviewedPolicyHandler(PolicyExternalToolHandler):
    """``PolicyExternalToolHandler`` with a third answer: ask a reviewer.

    The base class fires the BEFORE_TOOL_USE hooks, applies the policy, and
    forwards each result to ON_TOOL_CALL with its ``is_error``. This subclass
    adds one step: a call to a tool in ``ask`` that got that far still waits
    for the reviewer.
    """

    def __init__(self, policy: ToolPolicy, *, ask: list[str], reviewer: Reviewer) -> None:
        super().__init__(policy=policy)
        self._ask = ask
        self._reviewer = reviewer

    async def process_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        tool_call_id: str = "",
        job_id: str | None = None,
        session_id: str | None = None,
        tenant_id: str | None = None,
        room_id: str | None = None,
    ) -> ToolDecision:
        decision = await super().process_tool_call(
            tool_name,
            tool_input,
            tool_call_id=tool_call_id,
            job_id=job_id,
            session_id=session_id,
            tenant_id=tenant_id,
            room_id=room_id,
        )
        if not decision.approved or not any(fnmatch(tool_name, p) for p in self._ask):
            return decision
        # The reviewer sees the call as it would run, after any hook rewrite.
        arguments = tool_input if decision.modified_input is None else decision.modified_input
        if await self._reviewer(tool_name, arguments):
            return decision
        return ToolDecision(approved=False, reason=f"Tool '{tool_name}' rejected by the reviewer")


async def scripted_reviewer(tool_name: str, arguments: dict[str, Any]) -> bool:
    """Stands in for a human: approves writes under notes/, nothing else."""
    approved = str(arguments.get("path", "")).startswith("notes/")
    verdict = "approves" if approved else "rejects"
    print(f"   [ask] reviewer {verdict} {tool_name} {_show(arguments)}")
    return approved


class SimulatedAgent:
    """An external agent that runs its own tools, against an in-memory workspace.

    It knows the handler only through the ``ExternalToolHandler`` contract:
    ask before each call, run only an approved one, with the arguments the
    decision carries, and report every result with its error flag.
    """

    def __init__(self, handler: ExternalToolHandler, room_id: str) -> None:
        self._handler = handler
        self._room_id = room_id
        self._files = {
            "README.md": "# demo\nA small project.",
            "src/app.py": "# TODO: validate input\n# TODO: add logging\n# TODO: write tests",
            ".env": "DB_PASSWORD=example-only",
        }
        self.written: list[str] = []

    async def run(self, plan: list[tuple[str, dict[str, Any]]]) -> None:
        for number, (name, arguments) in enumerate(plan, start=1):
            print(f"\n{number}. agent proposes {name} {_show(arguments)}")
            await self._attempt(f"call-{number}", name, arguments)

    async def _attempt(self, call_id: str, name: str, arguments: dict[str, Any]) -> None:
        decision = await self._handler.process_tool_call(
            name, arguments, tool_call_id=call_id, room_id=self._room_id
        )
        if not decision.approved:
            print(f"   agent: refused, not run — {decision.reason}")
            return
        if decision.modified_input is None:
            print("   agent: approved, runs it")
        else:
            arguments = decision.modified_input
            print(f"   agent: approved, runs it with {_show(arguments)}")
        result, is_error = self._execute(name, arguments)
        await self._handler.on_tool_result(
            name,
            arguments,
            result,
            is_error=is_error,
            tool_call_id=call_id,
            room_id=self._room_id,
        )

    def _execute(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Run one tool: its result, and whether it failed."""
        if name == "Read":
            path = arguments["path"]
            if path not in self._files:
                return f"No such file: {path}", True
            return self._files[path], False
        if name == "Grep":
            hits = [
                f"{path}: {line}"
                for path, text in self._files.items()
                for line in text.splitlines()
                if arguments["pattern"] in line
            ]
            return "\n".join(hits[: arguments.get("max_results")]), False
        if name == "Write":
            self._files[arguments["path"]] = arguments["content"]
            self.written.append(arguments["path"])
            return f"Wrote {len(arguments['content'])} bytes to {arguments['path']}", False
        return f"Unknown tool: {name}", True


def install_hooks(kit: RoomKit) -> None:
    @kit.hook(HookTrigger.BEFORE_TOOL_USE, name="no-secrets")
    async def no_secrets(event: ToolCallEvent, _ctx: RoomContext) -> HookResult:
        if not str(event.arguments.get("path", "")).endswith(".env"):
            return HookResult.allow()
        print("   [BEFORE_TOOL_USE] no-secrets blocks the call")
        return HookResult.block("secret files stay out of the agent's reach")

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, name="cap-search")
    async def cap_search(event: ToolCallEvent, _ctx: RoomContext) -> HookResult:
        if event.name != "Grep" or "max_results" in event.arguments:
            return HookResult.allow()
        capped = {**event.arguments, "max_results": 2}
        print(f"   [BEFORE_TOOL_USE] cap-search rewrites the arguments to {_show(capped)}")
        # Replacement arguments ride metadata["arguments"]; the handler hands
        # them to the agent as ToolDecision.modified_input.
        return HookResult(action="allow", metadata={"arguments": capped})

    # A report: the agent already holds the result, so what this returns is
    # discarded. is_error is the agent's own verdict on the call.
    @kit.hook(HookTrigger.ON_TOOL_CALL, name="report")
    async def report(event: ToolCallEvent, _ctx: RoomContext) -> HookResult:
        outcome = "failed" if event.is_error else "ok"
        preview = str(event.result).replace("\n", " | ")
        print(
            f"   [ON_TOOL_CALL] {event.name} {_show(event.arguments)} {outcome} "
            f"(is_error={event.is_error}): {preview}"
        )
        return HookResult.allow()


async def main() -> None:
    setup_logging("external_tool_handler")

    kit = RoomKit()
    handler = ReviewedPolicyHandler(POLICY, ask=ASK, reviewer=scripted_reviewer)
    # Registering the channel that carries the handler is what wires the kit's
    # BEFORE_TOOL_USE and ON_TOOL_CALL hooks into it: ACPChannel for an ACP
    # agent, AIChannel (without a tool_handler) for a provider that runs its own
    # tools. This channel is never attached and never generates: the simulated
    # agent calls the handler itself, as ACPChannel does for a real agent.
    kit.register_channel(
        AIChannel("coding-agent", provider=MockAIProvider(), external_tool_handler=handler)
    )
    install_hooks(kit)
    # Tool hooks run in a room: the agent names it on every call.
    await kit.create_room(room_id=ROOM_ID)

    agent = SimulatedAgent(handler, ROOM_ID)
    try:
        await agent.run(PLAN)
    finally:
        await kit.close()
    print(f"\nFiles the agent wrote: {agent.written}")


if __name__ == "__main__":
    asyncio.run(main())
