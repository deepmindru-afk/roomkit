"""The shared conformance suite, run on the Python core (RFC §1.4, §26.1).

The scenarios come from roomkit-specs ``conformance/``, copied here by ``make
conformance-sync``, which records the specs commit in ``SPECS_COMMIT``; the
specs' checker validates them against their schema there. A
scenario this implementation is known to fail is listed in
``KNOWN_FAILURES`` with what it breaks; it is expected to fail, and passes
the suite only while it does, so a fix shows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.conformance._harness import Harness

HERE = Path(__file__).parent
SCENARIOS = HERE / "scenarios"

# Scenario id → the rule this implementation breaks (roomkit-specs
# conformance/README.md, "Known failures").
KNOWN_FAILURES: dict[str, str] = {
    "inbound/an-agent-s-output-updates-the-room": (
        "RFC §10.1 step 15: the room metadata updates of the channel outputs are "
        "collected (BroadcastResult.metadata_updates) and never applied to the room"
    ),
    "injection/an-injection-from-an-unregistered-channel-raises": (
        "RFC §10.5: send_event() from a channel no one registered raises "
        "ChannelNotFoundError, as for a channel the room does not bind, not "
        "ChannelNotRegisteredError"
    ),
}


def _scenarios() -> list[Path]:
    return sorted(SCENARIOS.glob("**/*.json"))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cases() -> list[Any]:
    cases = []
    for path in _scenarios():
        scenario_id = path.relative_to(SCENARIOS).with_suffix("").as_posix()
        marks = []
        if scenario_id in KNOWN_FAILURES:
            marks.append(pytest.mark.xfail(reason=KNOWN_FAILURES[scenario_id], strict=True))
        cases.append(pytest.param(path, id=scenario_id, marks=marks))
    return cases


def test_the_suite_is_synced_with_its_specs_commit() -> None:
    commit = (HERE / "SPECS_COMMIT").read_text(encoding="utf-8").strip()
    assert len(commit) == 40
    assert _scenarios()


def test_every_known_failure_names_a_scenario() -> None:
    ids = {path.relative_to(SCENARIOS).with_suffix("").as_posix() for path in _scenarios()}
    assert set(KNOWN_FAILURES) <= ids


@pytest.mark.parametrize("path", _cases())
async def test_scenario(path: Path) -> None:
    scenario = _load(path)
    assert scenario["id"] == path.relative_to(SCENARIOS).with_suffix("").as_posix()
    problems = await Harness(scenario).run()
    assert not problems, "\n".join(problems)
