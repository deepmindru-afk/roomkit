"""Names a package exports without importing their module at load.

A package lists such a name in its ``__all__`` and resolves it through the
``__getattr__`` this module builds (PEP 562): the defining module is imported
on first use, so ``import roomkit`` loads no optional dependency and no
provider module it did not load before. The package also imports each name
under ``TYPE_CHECKING``, so a type checker reads the real class.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping


def lazy_exports(package: str, exports: Mapping[str, str]) -> Callable[[str], object]:
    """A module ``__getattr__`` for ``package`` that resolves each name of
    ``exports`` from its module on first use.

    ``exports`` maps a public name to its defining module, or to
    ``"module:attribute"`` when the package exports the attribute under
    another name. A missing optional dependency raises the module's own
    ``ModuleNotFoundError``, naming the package to install.
    """

    def resolve(name: str) -> object:
        target = exports.get(name)
        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        module, _, attribute = target.partition(":")
        return getattr(importlib.import_module(module), attribute or name)

    return resolve
