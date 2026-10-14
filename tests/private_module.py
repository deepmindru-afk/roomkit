"""Load a module against fake dependencies without touching the shared one.

Reloading the shared module inside ``patch.dict(sys.modules, fakes)`` leaves
the fakes bound in its globals once the patch is gone, and every function of
the module reads those globals, whichever test imported it: a later test of
the same module then runs against the fakes.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType
from unittest.mock import patch


def load_private_copy(name: str, fakes: dict[str, object]) -> ModuleType:
    """A fresh copy of module *name*, run with *fakes* in ``sys.modules``.

    The copy is registered under *name* only while it runs, as an import
    would: ``dataclasses`` reads a class's module there to resolve its string
    annotations. Leaving the patch puts the shared module back, so it, and
    every test using it, keep their real dependencies.
    """
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.loader is not None, name
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fakes):
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module
