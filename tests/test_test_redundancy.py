"""The pruning tool of RMK-920 (``scripts/test_redundancy.py``).

Every pruning batch is accepted on ``diff`` naming no line a deleted test took
away, so a ``diff`` that read nothing would pass any deletion; and candidates
are read per test function, so a parametrized test is one candidate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from coverage import CoverageData

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "test_redundancy.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_redundancy", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(directory: Path, checkout: str, reached: dict[str, list[int]]) -> Path:
    """Write a run directory whose ``.coverage`` holds one context per key."""
    directory.mkdir()
    data = CoverageData(basename=str(directory / ".coverage"))
    for context, lines in reached.items():
        data.set_context(context)
        data.add_lines({f"{checkout}/src/roomkit/core.py": lines})
    data.write()
    return directory


def test_diff_fails_on_a_line_only_a_deleted_test_reached(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """And tells it from a line a race let a remaining test miss this time."""
    script = _load_script()
    before = _run(
        tmp_path / "before",
        "/one",
        {"tests/test_a.py::test_kept|run": [1, 5], "tests/test_a.py::test_gone|run": [1, 2, 3]},
    )
    after = _run(tmp_path / "after", "/two", {"tests/test_a.py::test_kept|run": [1]})

    assert script.diff(before, after) == 1
    gone, missed = capsys.readouterr().out.split("-- 1 lines not reached")
    assert "src/roomkit/core.py: 2-3" in gone
    assert "src/roomkit/core.py: 5" in missed
    again = _run(
        tmp_path / "again",
        "/two",
        {"tests/test_a.py::test_kept|run": [1], "tests/test_a.py::test_gone|run": [1, 2, 3]},
    )
    assert script.diff(before, again) == 0


def test_candidates_are_test_functions_with_their_phases_merged(tmp_path: Path) -> None:
    script = _load_script()
    run = _run(
        tmp_path / "run",
        "/one",
        {
            "tests/test_a.py::test_x[one]|run": [1, 2],
            "tests/test_a.py::test_x[two]|run": [1, 2],
            "tests/test_a.py::test_y|run": [1, 2],
            "tests/test_a.py::test_z|setup": [4],
            "tests/test_a.py::test_z|run": [1, 3],
        },
    )

    reached = script.read_coverage(run / ".coverage")

    assert script.identical_groups(reached) == [
        ["tests/test_a.py::test_x", "tests/test_a.py::test_y"]
    ]
    assert script.exclusive_lines(reached)["tests/test_a.py::test_z"] == 2
