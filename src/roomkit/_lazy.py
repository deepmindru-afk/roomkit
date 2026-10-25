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

import importlib
import sys
from collections.abc import Callable, Mapping


def lazy_exports(
    package: str, exports: Mapping[str, str]
) -> tuple[Callable[[str], object], Callable[[], list[str]]]:
    """A module ``__getattr__`` and ``__dir__`` for ``package``.

    ``__getattr__`` resolves each name of ``exports`` from its module on first
    use; ``__dir__`` lists them beside the module's own names, so ``dir()``
    and ``help()`` show them. ``exports`` maps a public name to its defining
    module, or to ``"module:attribute"`` when the package exports the
    attribute under another name. A missing optional dependency raises the
    module's own ``ModuleNotFoundError`` (``No module named 'httpx'``).
    """

    def resolve(name: str) -> object:
        target = exports.get(name)
        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        module, _, attribute = target.partition(":")
        return getattr(importlib.import_module(module), attribute or name)

    def listing() -> list[str]:
        return sorted({*vars(sys.modules[package]), *exports})

    return resolve, listing
