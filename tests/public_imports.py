"""Whether documents and examples import roomkit names from public paths.

A name is public where the ``__all__`` of ``roomkit`` or of one of its
packages (an ``__init__.py``) lists it, imported from that package. A module
path (``roomkit.models.event``) is internal even when it holds a public name,
and an underscore name is internal everywhere. A package itself is public:
``import roomkit.voice.stt`` and ``from roomkit.voice import stt`` follow the
rule, ``import roomkit.voice.stt.deepgram`` does not.

``check(paths)`` reads every line of the given files that starts a ``from
roomkit... import`` or ``import roomkit...`` statement, in code, in a
docstring, at a doctest prompt or in a fenced block at any indent, and every
inline code span that holds one; ``check(paths, docstrings=True)`` reads only
the triple-quoted strings of Python files, as the API reference renders them
from roomkit's own sources. It returns one finding per import that names a
non-public path, or that does not resolve. ``__all__`` is read from the source, so a name a
package exports only when its optional dependency is installed counts as
public; whether a name resolves is checked by importing it, and a name whose
optional dependency the environment lacks is left out.

``test_public_imports.py`` runs it on roomkit's examples, README, AGENTS.md,
``llms.txt``, ``docs/c7`` and the docstrings under ``src/roomkit``; ``make
docs`` runs it on the docs site's pages::

    uv run python -m tests.public_imports ../roomkit-docs/docs
"""

from __future__ import annotations

import ast
import importlib
import re
import sys
import textwrap
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import roomkit

_PACKAGE_ROOT = Path(roomkit.__file__).parent
_INLINE = re.compile(r"`((?:from|import)[ \t]+roomkit[^`\n]*)`")
_STATEMENT = re.compile(
    r"^[ \t]*(?:>>>[ \t]+)?"
    r"(?P<statement>from[ \t]+roomkit[\w.]*[ \t]+import[ \t]+(?:\([^)]*\)|[^\n]*)"
    r"|import[ \t]+roomkit[^\n]*)",
    re.M,
)
# roomkit's sources write every docstring with triple double quotes.
_TRIPLE_QUOTED = re.compile(r'"""(.*?)"""', re.S)
_SUFFIXES = (".py", ".md", ".txt")


@dataclass(frozen=True)
class Finding:
    """One import a reader should not copy."""

    path: str
    line: int
    statement: str
    problem: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.statement}: {self.problem}"


def check(
    paths: Iterable[Path], *, root: Path | None = None, docstrings: bool = False
) -> list[Finding]:
    """The imports in ``paths`` (files, or directories read recursively) that
    break the rule or do not resolve; ``root`` shortens the paths reported, and
    ``docstrings`` reads a Python file's triple-quoted strings only."""
    found: list[Finding] = []
    for path in _files(paths):
        shown = str(path.relative_to(root)) if root else str(path)
        for line, module, name in _imports(path, docstrings=docstrings):
            problem = _problem(module, name)
            if problem:
                statement = f"from {module} import {name}" if name else f"import {module}"
                found.append(Finding(shown, line, statement, problem))
    return found


def _files(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix in _SUFFIXES:
                    yield child
        elif path.is_file():
            yield path


def _imports(path: Path, *, docstrings: bool) -> Iterator[tuple[int, str, str | None]]:
    """``(line, module, name)`` per imported roomkit name; ``name`` is None for
    ``import roomkit.x``."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix != ".py":
        yield from _read(text, 0, spans=True)
    elif not docstrings:
        yield from _read(text, 0, spans=False)
    else:
        for string in _TRIPLE_QUOTED.finditer(text):
            yield from _read(string.group(1), text.count("\n", 0, string.start(1)), spans=True)


def _read(text: str, offset: int, *, spans: bool) -> Iterator[tuple[int, str, str | None]]:
    """The imports ``text`` holds, its first line at ``offset``: each statement
    line, and with ``spans`` each inline code span."""
    for statement in _STATEMENT.finditer(text):
        line = offset + text.count("\n", 0, statement.start())
        yield from _code_imports(statement.group("statement"), line)
    if spans:
        for span in _INLINE.finditer(text):
            yield from _code_imports(span.group(1), offset + text.count("\n", 0, span.start()))


def _code_imports(code: str, offset: int) -> Iterator[tuple[int, str, str | None]]:
    """The roomkit names one statement imports, with its line; a statement
    that is not valid Python (pseudo-code in a document) yields none."""
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and _is_roomkit(node.module):
            for alias in node.names:
                yield offset + node.lineno, node.module or "", alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_roomkit(alias.name):
                    yield offset + node.lineno, alias.name, None


def _is_roomkit(module: str | None) -> bool:
    return module is not None and (module == "roomkit" or module.startswith("roomkit."))


def _problem(module: str, name: str | None) -> str | None:
    """What is wrong with importing ``name`` from ``module``, or None."""
    if name is None:
        if not _is_package(module):
            return "a module path is internal; import the names its package exports"
        return _unresolved(module, None)
    if _is_package(f"{module}.{name}") or (name == "*" and _is_package(module)):
        return _unresolved(module, None)
    if name.startswith("_"):
        return "an underscore name is internal"
    if name not in exported(module):
        elsewhere = public_paths(name)
        hint = f"; public as {', '.join(elsewhere)}" if elsewhere else "; exported nowhere"
        unresolved = _unresolved(module, name)
        return f"{unresolved}{hint}" if unresolved else f"not public at {module}{hint}"
    return _unresolved(module, name)


def _unresolved(module: str, name: str | None) -> str | None:
    """Why ``module`` (or ``name`` on it) does not import, or None when it
    does or needs an optional dependency the environment lacks."""
    try:
        loaded = importlib.import_module(module)
    except ModuleNotFoundError as missing:
        if (missing.name or "").startswith("roomkit"):
            return f"does not resolve: no module {missing.name}"
        return None
    except ImportError:
        return None
    if name is None:
        return None
    try:
        getattr(loaded, name)
    except ModuleNotFoundError as missing:
        if (missing.name or "").startswith("roomkit"):
            return f"does not resolve: no module {missing.name}"
        return None
    except AttributeError:
        # A conditional export (``__all__ += [...]`` once its dependency
        # imports) is absent when that dependency is.
        if name in exported(module) and name not in getattr(loaded, "__all__", ()):
            return None
        return f"does not resolve: {module} has no {name}"
    return None


@cache
def _packages() -> dict[str, frozenset[str]]:
    """Each roomkit package and the names its ``__all__`` lists, read from
    the source, conditional additions included."""
    packages: dict[str, frozenset[str]] = {}
    for init in sorted(_PACKAGE_ROOT.rglob("__init__.py")):
        package = ".".join(("roomkit", *init.parent.relative_to(_PACKAGE_ROOT).parts))
        packages[package] = frozenset(_declared_all(init))
    return packages


def _declared_all(init: Path) -> Iterator[str]:
    tree = ast.parse(init.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            targets = [node.target]
        value = getattr(node, "value", None)
        names_all = any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets)
        if value is not None and names_all:
            yield from ast.literal_eval(value)


def _is_package(module: str) -> bool:
    return module in _packages()


def exported(package: str) -> frozenset[str]:
    """The names ``package``'s ``__all__`` lists (empty for a module)."""
    return _packages().get(package, frozenset())


def public_paths(name: str) -> list[str]:
    """Every public path of a name called ``name``."""
    return [f"{p}.{name}" for p, names in _packages().items() if name in names]


def main(argv: list[str]) -> int:
    found = check([Path(arg) for arg in argv])
    for finding in found:
        sys.stderr.write(f"{finding}\n")
    if found:
        sys.stderr.write(
            f"{len(found)} imports of roomkit names from a non-public path or that do not "
            "resolve: import each name from a package whose __all__ lists it.\n"
        )
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
