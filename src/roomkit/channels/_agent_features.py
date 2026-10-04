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

    carried: Callable[[AIChannel, bool], bool]
    """Whether an agent carries it, its own tool loop running or not."""


def _carries_skills(agent: AIChannel, runs_its_loop: bool) -> bool:
    """A registry offering skills; any registry when the agent's own loop
    runs, since one filled later would be served by that loop, outside the
    voice channel's gate."""
    skills = agent._skills
    return skills is not None and (runs_its_loop or skills.has_entries)


_FEATURES = (
    UnservedFeature("skills", "skills=", _carries_skills),
    UnservedFeature(
        "a human-input handler",
        "human_input_handler=",
        lambda agent, _: agent._human_input is not None,
    ),
    UnservedFeature("planning", None, lambda agent, _: agent._planner is not None),
    UnservedFeature("a sandbox", None, lambda agent, _: agent._sandbox is not None),
    UnservedFeature(
        "an external tool handler",
        None,
        lambda agent, _: agent._external_tool_handler is not None,
    ),
)


def unserved_on_realtime(agent: AIChannel, *, runs_its_loop: bool) -> list[UnservedFeature]:
    """What *agent* carries that a realtime session would never serve for it.

    *runs_its_loop* when the agent's own tool loop runs beside the session (a
    reasoning backend's agent): what it carries is then judged by what that
    loop could serve, an empty skill registry included. A pipeline agent's
    loop never runs in a session.
    """
    return [feature for feature in _FEATURES if feature.carried(agent, runs_its_loop)]
