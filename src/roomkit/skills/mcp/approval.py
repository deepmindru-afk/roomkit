"""The host's approval of code a served skill would cause to run (RFC §24.7)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SkillExecutionRequest:
    """What a host is asked before a tool that runs code is served while a
    skill an MCP server serves is active in the conversation.

    Attributes:
        name: The skill's name in the registry (``hf:hf-cli``).
        label: The host's label for the server that serves it.
        uri: The URI of the skill's ``SKILL.md`` on that server.
        fingerprint: The SHA-256 of the skill's files' URIs and digests,
            sorted (:attr:`SkillEntry.fingerprint`): an approval holds for it
            only. ``None`` for a dynamic skill, which no approval can bind.
        room_id: The conversation the skill is active in.
        tool: The tool about to run code.
    """

    name: str
    label: str
    uri: str
    fingerprint: str | None
    room_id: str | None
    tool: str


ApproveExecution = Callable[[SkillExecutionRequest], Awaitable[bool]]
"""A host's answer to a :class:`SkillExecutionRequest`: ``True`` approves."""
