"""Sandbox Executor — plug a command executor into an AIChannel.

Demonstrates how to use the SandboxExecutor ABC with AIChannel.
When a sandbox is provided, RoomKit automatically injects the executor's
tools (here: file reading, listing, search, git, diff and bash). The AI
decides when to use them based on the conversation.

WARNING: the executor below is NOT a sandbox. It runs the commands the
model chooses, arbitrary bash included, directly on this machine, as your
user, without asking. It starts in a fresh temporary directory (deleted on
exit), but nothing stops a command from leaving it. Run it only where that
is acceptable. For real isolation, see ``roomkit-sandbox``, which runs the
same tools in a Docker/Kubernetes container (``examples/sandbox_docker.py``).

Uses CLIChannel for interactive exploration. Try asking:
  - "Clone the repo https://github.com/rtk-ai/rtk and show its README"
  - "List the files in the clone"
  - "Search it for TODO comments"
  - "Show me its git log"

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/sandbox_executor.py
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import log_tool_call, require_env

from roomkit import (
    ChannelCategory,
    CLIChannel,
    HookTrigger,
    RoomKit,
    SandboxExecutor,
    SandboxResult,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.sandbox.tools import (
    SANDBOX_BASH_SCHEMA,
    SANDBOX_DIFF_SCHEMA,
    SANDBOX_FIND_SCHEMA,
    SANDBOX_GIT_SCHEMA,
    SANDBOX_GREP_SCHEMA,
    SANDBOX_LS_SCHEMA,
    SANDBOX_READ_SCHEMA,
)

# Only the tools _build_command implements: announcing write, edit or delete
# would let the model call tools that do nothing.
IMPLEMENTED_TOOLS = [
    SANDBOX_READ_SCHEMA,
    SANDBOX_LS_SCHEMA,
    SANDBOX_GREP_SCHEMA,
    SANDBOX_FIND_SCHEMA,
    SANDBOX_GIT_SCHEMA,
    SANDBOX_DIFF_SCHEMA,
    SANDBOX_BASH_SCHEMA,
]
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300


class LocalSandboxExecutor(SandboxExecutor):
    """Example executor that runs commands on this host (NOT sandboxed).

    For production use, implement execution inside a Docker/Kubernetes
    container. See ``roomkit-sandbox`` for a ready-made solution.
    """

    def __init__(self, workdir: str) -> None:
        self._workdir = workdir

    async def execute(
        self, command: str, arguments: dict[str, Any] | None = None
    ) -> SandboxResult:
        args = arguments or {}
        try:
            cmd = self._build_command(command, args)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._workdir,
                start_new_session=True,  # one process group, killed whole on timeout
            )
        except Exception as exc:
            return SandboxResult(exit_code=1, error=str(exc))
        timeout = _timeout(args) if command == "bash" else DEFAULT_TIMEOUT
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            # Killing bash alone would leave its children holding the pipes open.
            os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
            return SandboxResult(exit_code=124, error=f"Timed out after {timeout} s")
        return SandboxResult(
            exit_code=proc.returncode or 0,
            output=stdout.decode(errors="replace"),
            error=stderr.decode(errors="replace"),
        )

    def tool_definitions(self) -> list[dict[str, Any]]:
        return IMPLEMENTED_TOOLS

    def _build_command(self, command: str, args: dict[str, Any]) -> list[str]:
        if command == "read":
            cmd = ["cat", "-n", args["path"]]
        elif command == "ls":
            cmd = ["ls", "-la", args.get("path", ".")]
        elif command == "grep":
            cmd = ["grep", "-rn", args["pattern"]]
            if args.get("path"):
                cmd.append(args["path"])
            else:
                cmd.append(".")
        elif command == "find":
            cmd = ["find", args.get("path", ".")]
            if args.get("name"):
                cmd.extend(["-name", args["name"]])
            if args.get("type"):
                cmd.extend(["-type", args["type"]])
        elif command == "git":
            cmd = ["git"] + shlex.split(args.get("args", "status"))
        elif command == "diff":
            cmd = ["diff", args["file_a"], args["file_b"]]
        elif command == "bash":
            cmd = ["bash", "-c", args["command"]]
        else:
            raise ValueError(f"Unknown command: {command}")
        return cmd


def _timeout(args: dict[str, Any]) -> int:
    """The bash tool's own ``timeout`` argument, within 1..MAX_TIMEOUT seconds."""
    try:
        requested = int(args.get("timeout") or DEFAULT_TIMEOUT)
    except (TypeError, ValueError):
        requested = DEFAULT_TIMEOUT
    return min(max(requested, 1), MAX_TIMEOUT)


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")
    with tempfile.TemporaryDirectory(prefix="roomkit-sandbox-") as workdir:
        await run(env["ANTHROPIC_API_KEY"], workdir)


async def run(api_key: str, workdir: str) -> None:
    # --- Create the executor: commands start in workdir, on this host ---
    sandbox = LocalSandboxExecutor(workdir=workdir)

    # --- Set up RoomKit ---
    kit = RoomKit()

    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=AnthropicAIProvider(AnthropicConfig(api_key=api_key, model="claude-opus-5")),
        system_prompt=(
            "You are a helpful developer assistant with shell access to a scratch "
            "directory on the user's machine. You can read files, search code, run "
            "git commands, and execute bash commands. Use these tools to help the "
            "user explore and understand code."
        ),
        sandbox=sandbox,
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    # Show sandbox tool invocations in the terminal
    @kit.hook(HookTrigger.ON_TOOL_CALL)
    async def show_tool_call(event, _ctx):
        return log_tool_call(event, label="sandbox")

    await kit.create_room(room_id="sandbox-room")
    await kit.attach_channel("sandbox-room", "cli")
    await kit.attach_channel("sandbox-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    tools = [t["name"] for t in sandbox.tool_definitions()]
    await cli.run(
        kit,
        room_id="sandbox-room",
        welcome=(
            "\nSandbox demo — the AI has access to: "
            + ", ".join(tools)
            + f"\nCommands run on this machine, as you, starting in {workdir}"
            + " (deleted on exit).\nAsk the AI to explore files, search code,"
            + " or run commands.\n"
        ),
    )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
