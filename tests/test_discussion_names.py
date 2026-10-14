"""The names a discussion reads in a message, and the silence token (RFC §19.7.5)."""

from __future__ import annotations

import time

import pytest

from roomkit.orchestration.strategies.discussion._names import (
    NAME_LIMIT,
    Names,
    SilentToken,
    is_handle,
    name_key,
    read_names,
)

AGENTS = {a: a for a in ("investigator", "sre", "dev", "comms")}
"""Each agent's handle and channel id: here, its channel id is its handle."""


def test_names_are_read_in_order_ignoring_case_and_the_speaker() -> None:
    text = "@SRE please check deploys, then @comms; cc @sre again"
    assert read_names(text, AGENTS, speaker="investigator") == Names(("sre", "comms"))
    assert read_names("@sre over to you", AGENTS, speaker="sre") == Names()


def test_a_name_needs_a_boundary_before_it_and_drops_a_final_dot() -> None:
    assert read_names("mail ops@example.com", AGENTS) == Names()
    assert read_names("thanks @dev.", AGENTS) == Names(("dev",))
    assert read_names("@dev.team is not dev", AGENTS) == Names()
    assert read_names("(@comms)", AGENTS) == Names(("comms",))


def test_all_names_every_agent_and_unknown_names_nobody() -> None:
    assert read_names("@all heads up", AGENTS, speaker="sre") == Names(
        ("investigator", "dev", "comms")
    )
    assert read_names("@nobody here", AGENTS) == Names()


def test_names_in_code_or_quotes_are_not_read() -> None:
    text = "see\n```\n@dev run this\n```\n> @sre said this\nok @comms"
    assert read_names(text, AGENTS) == Names(("comms",))


def test_a_hostile_text_is_read_in_linear_time() -> None:
    """Thousands of unclosed fences and blank lines must not stall the room."""
    hostile = "```\n" * 20_000 + "\n" * 20_000 + "> \n" * 20_000 + "@sre"
    started = time.monotonic()
    read_names(hostile, AGENTS)
    read_names("~~~ x\n" + "@sre " * 50_000, AGENTS)
    assert time.monotonic() - started < 1.0


def test_an_unclosed_fence_hides_what_follows() -> None:
    assert read_names("@dev before\n```\n@sre inside, never closed", AGENTS) == Names(("dev",))


def test_people_are_named_too_and_an_agent_wins_a_shared_name() -> None:
    people = ("ops", "Alice Martin", "sre")
    names = read_names("@ops and @AliceMartin, ask @sre", AGENTS, people=people)
    assert names == Names(agents=("sre",), people=("ops", "Alice Martin"))
    assert name_key("Alice Martin") == "AliceMartin"


@pytest.mark.parametrize("text", ["(silent)", " (Silent). ", "SILENT", "silent.", ""])
def test_the_silent_token_is_read_leniently(text: str) -> None:
    assert SilentToken().is_silent(text)


@pytest.mark.parametrize("text", ["I stay (silent) for now", "silently", "pass"])
def test_text_that_says_something_is_not_silent(text: str) -> None:
    assert not SilentToken().is_silent(text)


def test_the_start_of_a_streamed_answer_is_held_while_it_may_become_the_token() -> None:
    token = SilentToken()
    for start in ("", " ", "(", "(sil", "(silent", "(silent)", "(silent).", "Sil"):
        assert token.may_become(start), start
    for start in ("(silent) but", "Hello", "(s!"):
        assert not token.may_become(start), start


def test_a_custom_token_without_brackets() -> None:
    token = SilentToken("PASS")
    assert token.is_silent("pass.") and token.may_become("pa")
    assert not token.is_silent("(pass)")


def test_names_hold_letters_of_any_script_and_are_never_cut() -> None:
    names = read_names("@Hélène, which region? café@dev", {"dev": "dev"}, people=["Hélène"])
    assert names == Names((), ("Hélène",))
    assert name_key("Zoë O'Brien-Smith") == "ZoëOBrien-Smith"
    # Cut short, a name could be another person's whole one.
    assert len(name_key("x" * 500)) == 500 > NAME_LIMIT


def test_only_one_final_period_is_dropped() -> None:
    assert read_names("thanks @dev.", {"dev": "dev"}).agents == ("dev",)
    assert read_names("thanks @dev..", {"dev": "dev"}).agents == ()


LUGE = {"Jev": "agent:3f2a", "Juriste": "agent:9b1c"}
"""Agents whose channel ids cannot be names, addressed by their handles."""


def test_a_handle_names_its_agent_by_channel_id_and_the_channel_id_names_nobody() -> None:
    names = read_names("@juriste, check clause 4 with @Jev", LUGE, speaker="agent:3f2a")
    assert names == Names(("agent:9b1c",))
    # The channel id is no name: "@agent:9b1c" reads as "@agent".
    assert read_names("@agent:9b1c over to you", LUGE) == Names()
    assert read_names("@all heads up", LUGE, speaker="agent:9b1c") == Names(("agent:3f2a",))


def test_a_person_named_like_a_handle_is_the_agents_name() -> None:
    names = read_names("@Jev and @Alice", LUGE, people=["Jev", "Alice"])
    assert names == Names(agents=("agent:3f2a",), people=("Alice",))


@pytest.mark.parametrize("value", ["dev", "Jev", "Hélène", "sre-2", "a.b", "x" * NAME_LIMIT])
def test_a_handle_is_a_name(value: str) -> None:
    assert is_handle(value)


@pytest.mark.parametrize(
    "value",
    ["", "agent:3f2a", "two words", "dev.", "all", "ALL", "x" * (NAME_LIMIT + 1), "dev\n", 3],
)
def test_what_cannot_name_an_agent_is_no_handle(value: object) -> None:
    assert not is_handle(value)


def test_a_decomposed_name_is_no_handle_since_a_message_names_it_composed() -> None:
    assert is_handle("Hélène") and not is_handle("He\u0301le\u0300ne")
