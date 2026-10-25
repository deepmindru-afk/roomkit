"""Names a package exports without importing their module at load.

A package lists such a name in its ``__all__`` and resolves it through the
``__getattr__`` this module builds (PEP 562): the defining module is imported
on first use, so ``import roomkit`` loads no optional dependency and no
provider module it did not load before. The package imports each name under
``TYPE_CHECKING`` and binds ``__getattr__`` only outside it::

    if TYPE_CHECKING:
        from roomkit.voice.stt.deepgram import DeepgramSTTProvider
    else:
        __getattr__, __dir__ = lazy_exports(__name__, {...})

so a type checker reads the real class, and reports a name the package does
not export instead of typing it ``object``.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def lazy_exports(
    package: str, exports: Mapping[str, str]
) -> tuple[Callable[[str], object], Callable[[], list[str]]]:
    """A module ``__getattr__`` and ``__dir__`` for ``package``.

    ``__getattr__`` resolves each name of ``exports`` from its module on first
    use. ``__dir__`` lists the module's own names and each lazy name whose
    module would import, so ``dir()``, ``help()`` and ``inspect.getmembers()``
    show it and never import a module whose extra is missing. ``exports`` maps
    a public name to its defining module, or to ``"module:attribute"`` when the
    package exports the attribute under another name. A missing optional
    dependency raises the module's own ``ModuleNotFoundError`` (``No module
    named 'httpx'``).
    """

    def resolve(name: str) -> object:
        target = exports.get(name)
        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        module, _, attribute = target.partition(":")
        return getattr(importlib.import_module(module), attribute or name)

    def listing() -> list[str]:
        seen: dict[str, bool] = {}
        loadable = [n for n, t in exports.items() if _loadable(t.partition(":")[0], seen)]
        return sorted({*vars(sys.modules[package]), *loadable})

    return resolve, listing


def _loadable(module: str, seen: dict[str, bool]) -> bool:
    """Whether importing ``module`` would find every module its top level
    imports, read from roomkit's sources: nothing is imported to answer. A
    module outside roomkit counts when its top-level package is found."""
    if module in sys.modules:
        return sys.modules[module] is not None
    if module in seen:
        return seen[module]
    if module != "roomkit" and not module.startswith("roomkit."):
        top = module.partition(".")[0]
        if top in sys.modules:
            return sys.modules[top] is not None
        return importlib.util.find_spec(top) is not None
    source, is_package = _source(module)
    if source is None:
        return False
    seen[module] = True  # an import cycle reads as loadable
    package = module if is_package else module.rpartition(".")[0]
    tree = ast.parse(source.read_text(encoding="utf-8"))
    seen[module] = all(_loadable(name, seen) for name in _required_imports(tree, package))
    return seen[module]


def _source(module: str) -> tuple[Path | None, bool]:
    """The file a roomkit module is read from, and whether it is a package."""
    path = _ROOT.joinpath(*module.split(".")[1:])
    if (path / "__init__.py").is_file():
        return path / "__init__.py", True
    if path.with_suffix(".py").is_file():
        return path.with_suffix(".py"), False
    return None, False


def _required_imports(tree: ast.Module, package: str) -> Iterator[str]:
    """The modules a module's top level imports: its import statements, and
    those of a ``try`` whose handler raises again; one under ``TYPE_CHECKING``
    or with a fallback is not required."""
    for node in tree.body:
        statements: list[ast.stmt] = [node]
        if isinstance(node, ast.Try):
            reraises = any(
                isinstance(stmt, ast.Raise) for handler in node.handlers for stmt in handler.body
            )
            statements = node.body if reraises else []
        for statement in statements:
            if isinstance(statement, ast.Import):
                yield from (alias.name for alias in statement.names)
            elif isinstance(statement, ast.ImportFrom):
                yield _absolute(statement, package)


def _absolute(node: ast.ImportFrom, package: str) -> str:
    if not node.level:
        return node.module or ""
    base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
    return f"{base}.{node.module}" if node.module else base
