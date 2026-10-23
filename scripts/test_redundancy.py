"""List the test suite's pruning candidates per area, and check a pruning kept its lines.

    uv run python scripts/test_redundancy.py run /tmp/rk/before
    uv run python scripts/test_redundancy.py candidates /tmp/rk/before --area conference
    uv run python scripts/test_redundancy.py files conference
    uv run python scripts/test_redundancy.py diff /tmp/rk/before /tmp/rk/after

``run`` runs the suite on one worker per core into a directory: a coverage
context per test (``.coverage``), a junit report (``junit.xml``) and the digest
of every library and test file (``manifest.json``). Arguments after ``--`` go
to pytest. It measures every roomkit module, including the ones pyproject
leaves out of coverage because CI lacks their native dependencies: with them
left out, a test of such a module reads as reaching no library line, and
deleting it drops no line the gate can see.

``candidates`` prints, per area, the test functions whose covered library lines
are identical (a parametrized function counted as the union of its cases), the
tests that ran no library line (skipped ones excluded), and the tests of 0.1 s
or more, each with the lines no other test function covers. Identical lines are
where to look, not a verdict: two tests reaching the same lines often check
different things (a minimum and a maximum, two clock cases).

``files`` prints an area's test files, to time the area alone.

``diff`` prints the library lines the first run covered and the second did not,
in two lists, and exits 1 when the first is not empty, so a pruning batch names
each line it dropped:

- taken away: a test that reached the line is gone, failed, or sits in a file
  that changed (a test rewritten in place may reach less under the same name);
  a line reached only outside any test (at import) counts too, and every line
  does when a helper or a conftest under ``tests/`` changed;
- missed this time: every test that reached it passed again from the same
  file, so a race decided the path (a run-to-run difference, not a pruning's
  doing).

It exits 2 when the runs do not compare: a manifest is missing, or the library
changed and with it the line numbers. The gate sees lines, not branches, and a
skipped test reaches nothing: deleting one is judged by reading it.

An area is a set of roomkit modules and the test directories that mirror them.
A file at the root of ``tests/`` belongs to the area most of its ``roomkit``
imports name, ``roomkit`` itself and ``roomkit.models`` aside (the vocabulary
every area shares); ``core`` is every module no other area claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess  # nosec B404
import sys
import xml.etree.ElementTree as ET  # nosec B405 - the junit file is the run's own
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple

from coverage import CoverageData

REPO_ROOT = Path(__file__).resolve().parent.parent

# area -> (roomkit modules, test directories); core takes everything else.
AREAS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "providers": (
        ("roomkit.providers",),
        ("tests/test_providers", "tests/providers", "tests/text_conformance"),
    ),
    "voice": (
        ("roomkit.voice", "roomkit.channels.voice", "roomkit.channels.realtime_voice"),
        ("tests/voice",),
    ),
    "conference": (
        (
            "roomkit.conference",
            "roomkit.video",
            "roomkit.channels.conference",
            "roomkit.channels.video",
            "roomkit.channels.av",
            "roomkit.channels.realtime_av",
        ),
        ("tests/conference", "tests/video"),
    ),
    "orchestration": (
        ("roomkit.orchestration", "roomkit.tasks"),
        ("tests/orchestration", "tests/tasks"),
    ),
}
CORE = "core"
# Imported by every area: a vote for none of them (``roomkit`` itself too).
SHARED = ("roomkit.models",)
SLOW = 0.1
COVERAGE = ".coverage"
JUNIT = "junit.xml"
MANIFEST = "manifest.json"
IMPORT = re.compile(r"^\s*(?:from|import)\s+(roomkit(?:\.\w+)*)", re.MULTILINE)
PHASE = re.compile(r"\|(?:setup|run|teardown)$")

Line = tuple[str, int]


class Outcome(NamedTuple):
    """What the junit report says of one test."""

    seconds: float
    skipped: bool
    failed: bool


def function_of(test_id: str) -> str:
    """Return the test function a node id names, its parameters folded."""
    return test_id.split("[", 1)[0]


def area_of(test_file: str, root: Path = REPO_ROOT) -> str:
    """Return the area a test file belongs to, by its directory or its imports."""
    for area, (_, directories) in AREAS.items():
        if any(test_file.startswith(directory + "/") for directory in directories):
            return area
    if test_file.count("/") > 1 or not (root / test_file).is_file():
        return CORE  # a root file deleted since the run has no imports left to read
    named: Counter[str] = Counter()
    for module in IMPORT.findall((root / test_file).read_text(encoding="utf-8")):
        if module != "roomkit" and not _within(module, SHARED):
            named[_area_of_module(module)] += 1
    ranked = named.most_common(2)
    if not ranked or (len(ranked) == 2 and ranked[0][1] == ranked[1][1]):
        return CORE
    return ranked[0][0]


def _area_of_module(module: str) -> str:
    for area, (modules, _) in AREAS.items():
        if _within(module, modules):
            return area
    return CORE


def _within(module: str, packages: tuple[str, ...]) -> bool:
    return any(module == package or module.startswith(package + ".") for package in packages)


def library_path(measured: str) -> str:
    """Return a measured file's path from ``src/``, whatever checkout ran it."""
    marker = "/src/roomkit/"
    index = measured.find(marker)
    return measured[index + 1 :] if index >= 0 else measured


def read_coverage(path: Path) -> dict[str, set[Line]]:
    """Return the library lines each test reached, its phases merged."""
    reached: dict[str, set[Line]] = defaultdict(set)
    for line, tests in _tests_by_line(path):
        for test_id in tests:
            reached[test_id].add(line)
    return reached


def _tests_by_line(path: Path, files: set[str] | None = None) -> Iterator[tuple[Line, set[str]]]:
    """Yield each covered line of a run with the tests that reached it (at import: none)."""
    data = CoverageData(basename=str(path))
    data.read()
    for measured in data.measured_files():
        source = library_path(measured)
        if files is not None and source not in files:
            continue
        for number, contexts in data.contexts_by_lineno(measured).items():
            yield (source, number), {PHASE.sub("", c) for c in contexts if c}


def covered_lines(path: Path) -> set[Line]:
    """Return every library line a run covered, by any test or none."""
    data = CoverageData(basename=str(path))
    data.read()
    return {
        (library_path(measured), number)
        for measured in data.measured_files()
        for number in data.lines(measured) or ()
    }


def read_junit(path: Path, root: Path = REPO_ROOT) -> dict[str, Outcome]:
    """Return each test of a junit report by its node id."""
    outcomes: dict[str, Outcome] = {}
    for case in ET.parse(path).iter("testcase"):  # nosec B314 - the run's own report
        test_id = _node_id(case.get("classname", ""), case.get("name", ""), root)
        outcomes[test_id] = Outcome(
            seconds=float(case.get("time", 0)),
            skipped=case.find("skipped") is not None,
            failed=case.find("failure") is not None or case.find("error") is not None,
        )
    return outcomes


def _node_id(classname: str, name: str, root: Path) -> str:
    parts = classname.split(".")
    for cut in range(len(parts), 0, -1):
        test_file = "/".join(parts[:cut]) + ".py"
        if (root / test_file).is_file():
            return "::".join([test_file, *parts[cut:], name])
    # A file deleted since the run: its module is the last part named like one.
    cut = max(
        (i + 1 for i, part in enumerate(parts) if part.startswith("test_")), default=len(parts)
    )
    return "::".join(["/".join(parts[:cut]) + ".py", *parts[cut:], name])


def by_function(reached: dict[str, set[Line]]) -> dict[str, set[Line]]:
    """Return the lines each test function reached, the union of its cases."""
    functions: dict[str, set[Line]] = defaultdict(set)
    for test_id, lines in reached.items():
        functions[function_of(test_id)] |= lines
    return functions


def exclusive_lines(functions: dict[str, set[Line]]) -> Counter[str]:
    """Return, per test function, how many lines no other test function covers."""
    owners: dict[Line, set[str]] = defaultdict(set)
    for function, lines in functions.items():
        for line in lines:
            owners[line].add(function)
    return Counter(next(iter(names)) for names in owners.values() if len(names) == 1)


def identical_groups(functions: dict[str, set[Line]]) -> list[list[str]]:
    """Return the groups of two or more test functions reaching the same lines."""
    by_lines: dict[frozenset[Line], list[str]] = defaultdict(list)
    for function, lines in functions.items():
        by_lines[frozenset(lines)].append(function)
    groups = [sorted(names) for names in by_lines.values() if len(names) > 1]
    return sorted(groups, key=lambda group: (-len(group), group))


class Areas:
    """The area of each test file, read once."""

    def __init__(self, root: Path = REPO_ROOT) -> None:
        self._root = root
        self._files: dict[str, str] = {}

    def of(self, test_id: str) -> str:
        test_file = test_id.split("::", 1)[0]
        if test_file not in self._files:
            self._files[test_file] = area_of(test_file, self._root)
        return self._files[test_file]


def print_groups(
    groups: list[list[str]], functions: dict[str, set[Line]], alone: Counter[str]
) -> None:
    """Print the groups of functions reaching identical lines."""
    extra = sum(len(group) - 1 for group in groups)
    print(f"\n-- identical lines: {len(groups)} groups, {extra} functions beyond the first")
    for group in groups:
        print(f"   {len(functions[group[0]])} lines")
        for function in group:
            print(f"     {function}  (alone {alone[function]})")


def print_nothing(ran: dict[str, Outcome], reached: dict[str, set[Line]]) -> None:
    """Print the tests that ran and reached no library line."""
    nothing = sorted(test_id for test_id in ran if test_id not in reached)
    print(f"\n-- no library line: {len(nothing)}")
    for test_id in nothing:
        print(f"   {test_id}")


def print_slow(ran: dict[str, Outcome], alone: Counter[str]) -> None:
    """Print the tests of ``SLOW`` seconds or more, slowest first."""
    slow = sorted(
        ((o.seconds, test_id) for test_id, o in ran.items() if o.seconds >= SLOW), reverse=True
    )
    print(f"\n-- {SLOW} s or more: {len(slow)}, {sum(s for s, _ in slow):.1f} s")
    for seconds, test_id in slow:
        print(f"   {seconds:6.2f} s  {test_id}  (alone {alone[function_of(test_id)]})")


def run(directory: Path, pytest_args: list[str]) -> int:
    """Run the suite into ``directory``, every roomkit module measured; pytest's exit code."""
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        *(sys.executable, "-m", "pytest", "-q", "-n", "auto", "--dist", "worksteal"),
        *("--cov=roomkit", f"--cov-config={os.devnull}", "--cov-context=test", "--cov-report="),
        f"--junitxml={directory / JUNIT}",
        *pytest_args,
    ]
    env = {**os.environ, "COVERAGE_FILE": str(directory / COVERAGE)}
    write_manifest(directory)
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode  # nosec B603


def write_manifest(directory: Path, root: Path = REPO_ROOT) -> None:
    """Record the digest of every library and test file a run is made of."""
    sources = [*(root / "src" / "roomkit").rglob("*.py"), *(root / "tests").rglob("*.py")]
    digests = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(sources)
    }
    (directory / MANIFEST).write_text(json.dumps(digests, indent=0), encoding="utf-8")


def read_manifest(directory: Path) -> dict[str, str]:
    """Return a run's file digests, empty when the run wrote none."""
    path = directory / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def candidates(directory: Path, areas: list[str]) -> None:
    """Print the totals and the three lists of candidates of each area asked for."""
    reached = read_coverage(directory / COVERAGE)
    outcomes = read_junit(directory / JUNIT)
    functions = by_function(reached)
    alone = exclusive_lines(functions)
    groups = identical_groups(functions)
    where = Areas()
    for area in areas:
        ran = {t: o for t, o in outcomes.items() if not o.skipped and where.of(t) == area}
        print(f"== {area}: {len(ran)} tests, {sum(o.seconds for o in ran.values()):.1f} s")
        failed = sum(o.failed for o in ran.values())
        if failed:
            print(f"   {failed} failed: the lines they reached are not the suite's")
        print_groups([g for g in groups if any(where.of(f) == area for f in g)], functions, alone)
        print_nothing(ran, reached)
        print_slow(ran, alone)
        print()


def files(area: str, root: Path = REPO_ROOT) -> list[str]:
    """Return the test files of an area, from the tree rather than from a run."""
    found = (str(path.relative_to(root)) for path in (root / "tests").rglob("test_*.py"))
    return sorted(test_file for test_file in found if area_of(test_file, root) == area)


def diff(before: Path, after: Path, root: Path = REPO_ROOT) -> int:
    """Print the lines ``after`` no longer covers, by cause; 1 if a pruning took one away."""
    was, now = read_manifest(before), read_manifest(after)
    if not was or not now or _library(was) != _library(now):
        print("-- the runs do not compare: a manifest is missing or the library changed")
        return 2
    dropped = covered_lines(before / COVERAGE) - covered_lines(after / COVERAGE)
    owners = dict(_tests_by_line(before / COVERAGE, {source for source, _ in dropped}))
    unchanged = unchanged_tests(was, now, read_junit(after / JUNIT, root))
    lost = {line for line in dropped if not owners[line] or not owners[line] <= unchanged}
    print_lines("taken away with a test that is gone, failed or changed", lost)
    print_lines("missed this time by tests that passed again unchanged", dropped - lost)
    return 1 if lost else 0


def _library(manifest: dict[str, str]) -> dict[str, str]:
    return {path: digest for path, digest in manifest.items() if path.startswith("src/")}


def unchanged_tests(
    was: dict[str, str], now: dict[str, str], outcomes: dict[str, Outcome]
) -> set[str]:
    """Return the tests that passed in the second run from a file both runs share."""
    changed = {
        path
        for path in was.keys() | now.keys()
        if path.startswith("tests/") and was.get(path) != now.get(path)
    }
    if any(not Path(path).name.startswith("test_") for path in changed):
        return set()  # a helper or a conftest changed: any test may reach less
    return {
        test_id
        for test_id, outcome in outcomes.items()
        if not (outcome.failed or outcome.skipped) and test_id.split("::", 1)[0] not in changed
    }


def print_lines(title: str, lines: set[Line]) -> None:
    """Print a titled list of lines, one row per file, numbers as ranges."""
    by_file: dict[str, list[int]] = defaultdict(list)
    for source, number in lines:
        by_file[source].append(number)
    print(f"-- {len(lines)} lines {title}")
    for source in sorted(by_file):
        print(f"   {source}: {_ranges(sorted(by_file[source]))}")


def _ranges(numbers: list[int]) -> str:
    spans: list[list[int]] = []
    for number in numbers:
        if spans and number == spans[-1][1] + 1:
            spans[-1][1] = number
        else:
            spans.append([number, number])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


COMMANDS: dict[str, Callable[[argparse.Namespace], int | None]] = {
    "run": lambda args: run(args.directory, [a for a in args.pytest_args if a != "--"]),
    "candidates": lambda args: candidates(args.directory, args.area or [*AREAS, CORE]),
    "files": lambda args: print("\n".join(files(args.area))),
    "diff": lambda args: diff(args.before, args.after),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser("run", help="run the suite into a directory")
    running.add_argument("directory", type=Path)
    running.add_argument("pytest_args", nargs=argparse.REMAINDER)
    listing = commands.add_parser("candidates", help="list one or every area's candidates")
    listing.add_argument("directory", type=Path)
    listing.add_argument("--area", choices=[*AREAS, CORE], action="append")
    listing_files = commands.add_parser("files", help="list an area's test files")
    listing_files.add_argument("area", choices=[*AREAS, CORE])
    gate = commands.add_parser("diff", help="name the lines a pruning dropped")
    gate.add_argument("before", type=Path)
    gate.add_argument("after", type=Path)
    args = parser.parse_args(argv)
    return COMMANDS[args.command](args) or 0


if __name__ == "__main__":
    sys.exit(main())
