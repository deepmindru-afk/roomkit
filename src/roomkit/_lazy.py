"""Names a package exports without importing their module at load.

A package lists such a name in its ``__all__`` and resolves it through the
``__getattr__`` this module builds (PEP 562): the defining module is imported
on first use, so ``import roomkit`` loads no optional dependency and no
provider module it did not load before. The package imports each name under
``TYPE_CHECKING`` and binds ``__getattr__`` only outside it::

    if TYPE_CHECKING:
        from roomkit.voice.stt.deepgram import DeepgramSTTProvider
    else:
        __getattr__ = lazy_exports(__name__, {...})

so a type checker reads the real class, and reports a name the package does
not export instead of typing it ``object``. There is no ``__dir__``:
``dir()``, ``help()`` and ``inspect.getmembers()`` read the names the package
has bound and never import a lazy one, so a missing extra cannot fail them.
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
    ``ModuleNotFoundError`` (``No module named 'httpx'``).
    """

    def resolve(name: str) -> object:
        target = exports.get(name)
        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        module, _, attribute = target.partition(":")
        return getattr(importlib.import_module(module), attribute or name)

    return resolve
