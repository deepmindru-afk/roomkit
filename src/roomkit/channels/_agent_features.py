"""What an agent's own channel serves for it that a realtime session does not
(RFC §12.4.1, §19.5).

An ``AIChannel`` serves its skills, its human-input tools, its planner, its
sandbox's commands and an external tool handler's tools inside its own tool
loop. A realtime voice session serves none of an agent's: it declares the
voice channel's tools and serves the voice channel's skills and human-input
tools. An agent that carries one of these is refused where it would meet a
realtime session (a reasoning backend's agent, a realtime pipeline's agent),
rather than leave a model calling a tool it was never told of.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.channels.ai import AIChannel


@dataclass(frozen=True, slots=True)
class UnservedFeature:
    """One thing an agent carries that a realtime session never serves for it."""

    what: str
    """How a refusal names it."""

    instead: str | None
    """The voice channel's own option that serves it in its sessions, if any."""

    short: str
    """How a refusal names it after an agent's: ``an agent's planning``."""

    carried: Callable[[AIChannel], bool]
    """Whether an agent carries it."""


_FEATURES = (
    # Any registry, an empty one included: a skill added to it after the check
    # would open its gated tools with nothing to gate them.
    UnservedFeature("skills", "skills=", "skills", lambda agent: agent._skills is not None),
    UnservedFeature(
        "a human-input handler",
        "human_input_handler=",
        "human-input tools",
        lambda agent: agent._human_input.given,
    ),
    UnservedFeature("planning", None, "planning", lambda agent: agent._planner is not None),
    UnservedFeature(
        "a sandbox", None, "sandbox commands", lambda agent: agent._sandbox is not None
    ),
    UnservedFeature(
        "an external tool handler",
        None,
        "external tools",
        lambda agent: agent._external_tool_handler is not None,
    ),
)


def unserved_on_realtime(agent: AIChannel) -> list[UnservedFeature]:
    """What *agent* carries that a realtime session would never serve for it."""
    return [feature for feature in _FEATURES if feature.carried(agent)]
