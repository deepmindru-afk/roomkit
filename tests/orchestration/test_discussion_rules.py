"""How the room works, as a discussion's turns are told it (RFC §19.7.5 rule 14).

The notes of each turn end with the room's rules: the host's when the
discussion has them, else the default, which asks the agent to end its
message with the handle of the agent that must act next. The rules are part
of the stored configuration, so every process gives its turns with the same.
"""

from __future__ import annotations

import pytest

from roomkit import Agent, RoomKit
from roomkit.orchestration.strategies.discussion import Discussion
from roomkit.orchestration.strategies.discussion._config import CONFIG_KEY, DiscussionConfig
from roomkit.orchestration.strategies.discussion._notes import DEFAULT_RULES
from tests.orchestration.test_strategy_discussion import NOTES, _provider, _say, _settle, _texts
from tests.test_framework import SimpleChannel


def _pair(rules: str | None = None) -> tuple[Discussion, Agent, Agent]:
    jev = Agent("jev", provider=_provider("Clause 4 caps it."), name="Jev")
    counsel = Agent("counsel", provider=_provider("Lawful."), name="Counsel")
    return Discussion([jev, counsel], everyone=["jev"], rules=rules), jev, counsel


async def _notes_of(strategy: Discussion, agent: Agent) -> str:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    await kit.create_room(room_id="r1", orchestration=strategy)
    await kit.attach_channel("r1", "ops")
    await _say(kit, "What does clause 4 say?")
    await _settle(kit)
    await kit.close()
    return next(t for t in _texts(agent.provider) if NOTES in t)  # type: ignore[arg-type]


async def test_the_default_asks_for_the_handle_of_whoever_acts_next() -> None:
    strategy, jev, _ = _pair()

    notes = await _notes_of(strategy, jev)

    assert "end your message with its handle, as in @counsel" in notes
    assert "a name without @ reaches nobody" in notes
    assert '"your turn" or "over to you" alone, in any language' in notes
    assert "Answer exactly (silent)" in notes


async def test_the_hosts_rules_take_the_defaults_place() -> None:
    strategy, jev, _ = _pair("Hand over with {handle}. Nothing to add: {silent}. {kept}")

    notes = await _notes_of(strategy, jev)

    assert "Hand over with @counsel. Nothing to add: (silent). {kept}" in notes
    assert "a name without @ reaches nobody" not in notes
    # What the turn knows of the room stays the discussion's.
    assert "You are @jev," in notes and "- @counsel: Counsel" in notes


def test_the_rules_are_stored_with_the_discussion() -> None:
    strategy, _, _ = _pair("Hand over with {handle}.")

    stored = DiscussionConfig.of(strategy, default_depth=6)
    read = DiscussionConfig.stored({CONFIG_KEY: stored.model_dump()})

    assert read is not None and read.rules == "Hand over with {handle}."


def test_a_discussion_stored_before_rules_reads_with_the_default() -> None:
    strategy, _, _ = _pair()
    stored = DiscussionConfig.of(strategy, default_depth=6).model_dump(exclude={"rules"})

    read = DiscussionConfig.stored({CONFIG_KEY: stored})

    assert read is not None and read.rules is None
    assert "{handle}" in DEFAULT_RULES and "{silent}" in DEFAULT_RULES


def test_blank_rules_are_refused() -> None:
    with pytest.raises(ValueError, match="rules must not be blank"):
        _pair("   ")
