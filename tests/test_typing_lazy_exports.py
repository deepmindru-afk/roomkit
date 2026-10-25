"""A type checker reads a lazily exported name as what it is, and refuses one
that is not exported.

A package that exports a name without importing its module at load binds a
PEP 562 ``__getattr__`` (``roomkit._lazy``). A type checker that sees that
``__getattr__`` types any attribute as its return type, so a misspelled
import passed as ``object``. The packages bind it outside ``TYPE_CHECKING``
and import the real names under it: this runs ty on an application that
imports every lazily exported name, and one misspelled name per package.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import roomkit

_PACKAGE_ROOT = Path(roomkit.__file__).parent


def _packages_with_lazy_names() -> dict[str, list[str]]:
    """``{package: [lazy names]}``: the names a package's ``__all__`` lists
    and its namespace does not hold until first use."""
    found: dict[str, list[str]] = {}
    for init in sorted(_PACKAGE_ROOT.rglob("__init__.py"), key=str):
        package = ".".join(("roomkit", *init.parent.relative_to(_PACKAGE_ROOT).parts))
        try:
            module: ModuleType = importlib.import_module(package)
        except ImportError:
            continue
        if "__getattr__" not in vars(module):
            continue
        lazy = [name for name in getattr(module, "__all__", ()) if name not in vars(module)]
        if lazy:
            found[package] = lazy
    return found


@pytest.mark.typing  # ty in its own process: run by make typecheck
def test_ty_types_each_lazy_name_and_reports_a_misspelled_one(tmp_path: Path) -> None:
    packages = _packages_with_lazy_names()
    # The walk must reach the lazy packages, or this would check nothing.
    assert "roomkit.voice.stt" in packages and "roomkit.sources" in packages
    lines = ["from typing import reveal_type"]
    for index, (package, names) in enumerate(sorted(packages.items())):
        for position, name in enumerate(names):
            alias = f"name_{index}_{position}"
            lines += [f"from {package} import {name} as {alias}", f"reveal_type({alias})"]
        lines.append(f"from {package} import {names[0]}Misspelled as typo_{index}")
    application = tmp_path / "application.py"
    application.write_text("\n".join(lines) + "\n")

    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "ty"),
            "check",
            "--python",
            sys.executable,
            "--output-format",
            "concise",
            str(application),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr

    misspelled = [line for line in output.splitlines() if "Misspelled" in line]
    assert len(misspelled) == len(packages), output
    revealed = [line for line in output.splitlines() if "Revealed type" in line]
    assert len(revealed) == sum(len(names) for names in packages.values()), output
    untyped = ("Revealed type: `object`", "Revealed type: `Unknown`")
    assert [line for line in revealed if line.endswith(untyped)] == []
