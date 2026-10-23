"""The pruning tool (``scripts/test_redundancy.py``).

Every pruning batch is accepted on ``diff`` naming no line a pruning took away,
so a ``diff`` that read nothing, or that let a test rewritten in place lose a
line under its old name, would pass any deletion; and candidates are read per
test function, so a parametrized test is one candidate.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from coverage import CoverageData

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "test_redundancy.py"
_KEPT = "tests/test_a.py::test_kept"
_GONE = "tests/test_b.py::test_gone"

Pruning = Callable[[Path], tuple[dict[str, list[int]], frozenset[str]]]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_redundancy", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkout(root: Path) -> Path:
    """A library module and two test files, as a run would find them."""
    for name in ("src/roomkit/core.py", "tests/test_a.py", "tests/test_b.py"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("pass\n")
    return root


def _coverage(directory: Path, root: Path, reached: dict[str, list[int]]) -> None:
    data = CoverageData(basename=str(directory / ".coverage"))
    for context, lines in reached.items():
        data.set_context(context)
        data.add_lines({f"{root}/src/roomkit/core.py": lines})
    data.write()


def _run(
    script: ModuleType,
    directory: Path,
    root: Path,
    reached: dict[str, list[int]],
    failed: frozenset[str] = frozenset(),
) -> Path:
    """Write what ``run`` writes: a context per test (``""`` is import time),
    a junit report of the tests named, and the checkout's manifest."""
    directory.mkdir()
    _coverage(directory, root, {f"{t}|run" if t else "": lines for t, lines in reached.items()})
    cases = "".join(
        f'<testcase classname="{test_id.split("::")[0][:-3].replace("/", ".")}" '
        f'name="{test_id.split("::")[1]}">{"<failure/>" if test_id in failed else ""}</testcase>'
        for test_id in reached
        if test_id
    )
    (directory / "junit.xml").write_text(
        f"<testsuites><testsuite>{cases}</testsuite></testsuites>"
    )
    script.write_manifest(directory, root)
    return directory


def _deleted(root: Path) -> tuple[dict[str, list[int]], frozenset[str]]:
    (root / "tests/test_b.py").unlink()
    return {_KEPT: [1, 5], "": [9]}, frozenset()


def _rewritten_in_place(root: Path) -> tuple[dict[str, list[int]], frozenset[str]]:
    (root / "tests/test_a.py").write_text("pass  # rewritten\n")
    return {_KEPT: [1], _GONE: [1, 2], "": [9]}, frozenset()


def _failed(root: Path) -> tuple[dict[str, list[int]], frozenset[str]]:
    return {_KEPT: [1], _GONE: [1, 2], "": [9]}, frozenset({_KEPT})


def _imported_by_no_test(root: Path) -> tuple[dict[str, list[int]], frozenset[str]]:
    return {_KEPT: [1, 5], _GONE: [1, 2]}, frozenset()


@pytest.mark.parametrize(
    ("prune", "lost"),
    [(_deleted, "2"), (_rewritten_in_place, "5"), (_failed, "5"), (_imported_by_no_test, "9")],
)
def test_diff_fails_on_a_line_a_pruning_took_away(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], prune: Pruning, lost: str
) -> None:
    script = _load_script()
    root = _checkout(tmp_path / "checkout")
    before = _run(script, tmp_path / "before", root, {_KEPT: [1, 5], _GONE: [1, 2], "": [9]})
    reached, failed = prune(root)
    after = _run(script, tmp_path / "after", root, reached, failed)

    assert script.diff(before, after, root) == 1
    taken, _ = capsys.readouterr().out.split("-- 0 lines missed")
    assert f"src/roomkit/core.py: {lost}\n" in taken


def test_diff_passes_a_line_a_race_let_an_unchanged_test_miss(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_script()
    root = _checkout(tmp_path / "checkout")
    before = _run(script, tmp_path / "before", root, {_KEPT: [1, 5], _GONE: [1, 2]})
    after = _run(script, tmp_path / "after", root, {_KEPT: [1], _GONE: [1, 2]})

    assert script.diff(before, after, root) == 0
    assert "src/roomkit/core.py: 5" in capsys.readouterr().out.split("-- 1 lines missed")[1]


def test_diff_refuses_runs_of_two_libraries(tmp_path: Path) -> None:
    script = _load_script()
    root = _checkout(tmp_path / "checkout")
    before = _run(script, tmp_path / "before", root, {_KEPT: [1]})
    (root / "src/roomkit/core.py").write_text("pass  # moved\n")
    after = _run(script, tmp_path / "after", root, {_KEPT: [1]})

    assert script.diff(before, after, root) == 2


def test_candidates_are_test_functions_with_their_phases_merged(tmp_path: Path) -> None:
    script = _load_script()
    (tmp_path / "run").mkdir()
    _coverage(
        tmp_path / "run",
        tmp_path,
        {
            "tests/test_a.py::test_x[small]|run": [1],
            "tests/test_a.py::test_x[big]|run": [1, 2],
            "tests/test_a.py::test_y|run": [1, 2],
            "tests/test_a.py::test_z|setup": [4],
            "tests/test_a.py::test_z|run": [1, 3],
        },
    )

    functions = script.by_function(script.read_coverage(tmp_path / "run" / ".coverage"))

    assert script.identical_groups(functions) == [
        ["tests/test_a.py::test_x", "tests/test_a.py::test_y"]
    ]
    assert script.exclusive_lines(functions)["tests/test_a.py::test_z"] == 2
