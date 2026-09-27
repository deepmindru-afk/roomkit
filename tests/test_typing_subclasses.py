"""Public classes built from mixins stay subclassable under mypy.

``make typecheck`` runs ``ty``, which does not compare the bases of a class.
mypy does, but only in a class with more than one direct base: every name two
bases define must be compatible. So ``class App(ACPChannel)`` passed while
``class App(Mixin, ACPChannel)``, the shape an application uses to add
behaviour, failed from 0.91.0 on (RMK-224). This writes that shape for every
public class with several bases and runs mypy on it, as an application would.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

from mypy import api

import roomkit


def _public_classes_with_several_bases() -> list[str]:
    names = []
    for name in roomkit.__all__:
        obj = getattr(roomkit, name)
        if (
            inspect.isclass(obj)
            and obj.__module__.startswith("roomkit.")
            and len(obj.__bases__) > 1
        ):
            names.append(name)
    return sorted(names)


def test_a_two_base_subclass_of_each_public_mixin_class_passes_mypy(tmp_path: Path) -> None:
    names = _public_classes_with_several_bases()
    assert "ACPChannel" in names  # the one that regressed
    lines = [
        "from __future__ import annotations",
        f"from roomkit import {', '.join(names)}",
        "class Mixin: ...",
    ]
    lines += [f"class App{name}(Mixin, {name}): ..." for name in names]
    application = tmp_path / "application.py"
    application.write_text("\n".join(lines) + "\n")

    # No cache: a stale one reported an old verdict while this was diagnosed.
    report, errors, status = api.run(
        [str(application), "--no-incremental", "--cache-dir", os.devnull]
    )

    assert status == 0, report + errors
