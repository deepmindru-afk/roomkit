"""A realtime pipeline writes an agent's identity block only if the agent lets it.

The session's prompt is built at the install and rebuilt in a room's language;
an agent with ``identity_in_prompt=False`` keeps its prompt as written on both,
and one without the opt-out gets the block, as before.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from roomkit.channels.agent import Agent
from roomkit.models.room import Room
from roomkit.orchestration._realtime_pipeline import RealtimePipeline


def _pipeline(agent: Agent) -> RealtimePipeline:
    handler = MagicMock()
    handler.get_room_language.return_value = "French"
    return RealtimePipeline(
        MagicMock(),
        MagicMock(),
        [agent],
        [],
        handler,
        agent.channel_id,
        greet_on_handoff=False,
        greeting_prompt=None,
    )


def _agent(**kwargs: object) -> Agent:
    return Agent("triage", role="Triage", system_prompt="Route the caller.", **kwargs)  # type: ignore[arg-type]


def test_an_agent_that_opted_out_keeps_its_prompt_at_the_install_and_in_a_language() -> None:
    pipeline = _pipeline(_agent(identity_in_prompt=False))

    assert pipeline.agent_configs["triage"]["system_prompt"] == "Route the caller."
    assert pipeline._prompt_for("triage", Room(id="r1")) == "Route the caller."


def test_without_the_opt_out_the_block_is_written_on_both() -> None:
    pipeline = _pipeline(_agent())

    assert "--- Agent Identity ---" in pipeline.agent_configs["triage"]["system_prompt"]
    assert "Always respond in French" in (pipeline._prompt_for("triage", Room(id="r1")) or "")
