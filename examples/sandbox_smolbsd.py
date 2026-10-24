"""Sandbox with SmolBSD — VM-isolated code execution for local assistants.

Demonstrates using roomkit-sandbox with SmolBSD to give an AI agent
sandboxed command execution inside a NetBSD microVM: a kernel of its own,
booted in about half a second, driven over a VirtIO console (no SSH, no
port opened on the host).

The VM outlives this script: run it again and the agent finds the same VM,
with its /workspace and whatever it installed. ``--cleanup`` deletes the
VM and its workspace.

Prerequisites:
  - Linux with KVM, QEMU and socat
  - passt, with QEMU 10.1 or later: without it, the VM can reach the
    services listening on the host's localhost
  - SmolBSD cloned: git clone https://github.com/NetBSDfr/smolBSD.git
  - Builder image: cd smolBSD && bmake fetchimg
  - The roomkit-sandbox guest image, built from a roomkit-sandbox checkout:
      smolbsd/build.sh /path/to/smolBSD NOSUDO=1
  - roomkit-sandbox 0.5.0 or later (the run commands below install it)

Environment variables:
    ANTHROPIC_API_KEY — Anthropic API key
    SMOLBSD_DIR       — path to the smolBSD clone

Try asking:
  - "Show system information"
  - "Clone https://github.com/NetBSDfr/smolBSD and tell me what startnb.sh does"
  - "Write a shell script that counts the files per extension, then run it"
  - Quit, run again, and ask "What is in the workspace?"

Run with:
    ANTHROPIC_API_KEY=sk-... SMOLBSD_DIR=/path/to/smolBSD \\
        uv run --with "roomkit-sandbox>=0.5.0" python examples/sandbox_smolbsd.py

Delete the VM and its workspace:
    SMOLBSD_DIR=/path/to/smolBSD \\
        uv run --with "roomkit-sandbox>=0.5.0" python examples/sandbox_smolbsd.py --cleanup
"""

from __future__ import annotations

import asyncio
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import log_tool_call, require_env

from roomkit import AIChannel, ChannelCategory, CLIChannel, HookTrigger, RoomKit
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig

SESSION_ID = "example-smolbsd-sandbox"


def require_sandbox() -> None:
    """Exit unless roomkit-sandbox 0.5.0+ is installed.

    0.4.0 imports fine but runs the old SSH backend: its first command
    times out after 30 s.
    """
    try:
        installed = version("roomkit-sandbox")
    except PackageNotFoundError:
        installed = None
    match = re.match(r"(\d+)\.(\d+)", installed or "")
    if match is None or (int(match[1]), int(match[2])) < (0, 5):
        found = f"found {installed}" if installed else "not installed"
        sys.exit(
            f"Error: this example needs roomkit-sandbox 0.5.0 or later ({found}): "
            'uv run --with "roomkit-sandbox>=0.5.0" python examples/sandbox_smolbsd.py'
        )


async def cleanup(backend) -> None:
    """Delete this example's VM and its workspace disk."""
    for info in await backend.list_containers():
        if info.session_id == SESSION_ID:
            await backend.delete_container(info.container_id)
            print(f"Deleted VM {info.container_id} and its workspace.")


async def main() -> None:
    require_sandbox()
    from roomkit_sandbox import ContainerSandboxExecutor
    from roomkit_sandbox.commands import NativeCommandBuilder
    from roomkit_sandbox.smolbsd_backend import SmolBSDSandboxBackend

    if "--cleanup" in sys.argv:
        env = require_env("SMOLBSD_DIR")
        await cleanup(SmolBSDSandboxBackend(smolbsd_dir=env["SMOLBSD_DIR"]))
        return

    env = require_env("ANTHROPIC_API_KEY", "SMOLBSD_DIR")

    # One NetBSD microVM per session, booted from the roomkit-sandbox image
    backend = SmolBSDSandboxBackend(smolbsd_dir=env["SMOLBSD_DIR"])
    sandbox = ContainerSandboxExecutor(
        backend=backend,
        session_id=SESSION_ID,
        command_builder=NativeCommandBuilder(),  # No RTK on NetBSD, use native commands
    )

    kit = RoomKit()

    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=AnthropicAIProvider(
            AnthropicConfig(api_key=env["ANTHROPIC_API_KEY"], model="claude-opus-5")
        ),
        system_prompt=(
            "You are a helpful developer assistant with a sandboxed environment: "
            "a NetBSD microVM with git, curl, bash and jq, and outbound network "
            "access. You can read, write and edit files, search code, run git and "
            "execute shell commands there. Work in /workspace: it is kept between "
            "conversations."
        ),
        sandbox=sandbox,
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    @kit.hook(HookTrigger.ON_TOOL_CALL)
    async def show_tool_call(event, _ctx):
        return log_tool_call(event, label="smolbsd")

    await kit.create_room(room_id="sandbox-room")
    await kit.attach_channel("sandbox-room", "cli")
    await kit.attach_channel("sandbox-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    tools = [t["name"] for t in sandbox.tool_definitions()]
    await cli.run(
        kit,
        room_id="sandbox-room",
        welcome=(
            "\nSmolBSD Sandbox demo — tools: "
            + ", ".join(tools)
            + "\nCommands run inside a NetBSD microVM with true VM isolation.\n"
        ),
    )

    # The VM keeps running: the next run picks it up with its workspace
    await sandbox.close()
    await kit.close()
    if await backend.find_container(SESSION_ID):
        print("\nThe VM keeps running with its /workspace; --cleanup deletes them.")


if __name__ == "__main__":
    asyncio.run(main())
