"""Shared sandbox tool handlers used by AIChannel."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.sandbox.tools import SANDBOX_TOOL_PREFIX

if TYPE_CHECKING:
    from roomkit.sandbox.executor import SandboxExecutor

logger = logging.getLogger("roomkit.channels.sandbox")


async def handle_sandbox_command(
    tool_name: str,
    arguments: dict[str, Any],
    executor: SandboxExecutor,
) -> str:
    """Route a sandbox tool call to the executor.

    Strips the ``sandbox_`` prefix and delegates to
    :meth:`SandboxExecutor.execute`.

    Returns:
        JSON-encoded :class:`SandboxResult`.
    """
    command = tool_name.removeprefix(SANDBOX_TOOL_PREFIX)
    # A failure propagates: the channel reads it as any raised call, with its
    # marker, the class for the model and the message for the observers
    # (RFC §9.3).
    result = await executor.execute(command, arguments)
    return result.model_dump_json()
