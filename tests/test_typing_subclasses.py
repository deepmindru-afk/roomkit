"""Public classes built from several bases stay subclassable under mypy.

``make typecheck`` runs ``ty``, which does not compare the bases of a class.
mypy does, but only in a class with more than one direct base: every name two
bases define must be compatible in MRO order. So ``class App(ACPChannel)`` can
pass while ``class App(Mixin, ACPChannel)``, the shape an application uses to
add behaviour, fails. This writes that shape for every public class with
several bases, in every public module, and runs mypy on it the way an
application would: in its own process, with no configuration and no cache.
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import roomkit

# Known incompatible, and not by annotation: each is a VoiceBackend and a
# VideoBackend at once, and the two ABCs disagree on ``accept``,
# ``capabilities``, ``get_session``, ``list_sessions`` and ``connect``.
_BASES_THAT_DISAGREE = {
    "roomkit.video.backends.fastrtc.FastRTCVideoBackend",
    "roomkit.video.backends.rtp.RTPVideoBackend",
    "roomkit.video.backends.sip.SIPVideoBackend",
}


def _public_classes_with_several_bases() -> dict[str, str]:
    """``{qualified name: module}`` for every such class a public module defines.

    A module whose optional dependency is not installed cannot be imported,
    so it cannot be subclassed here either: it is skipped.
    """
    found: dict[str, str] = {}
    for module_info in pkgutil.walk_packages(roomkit.__path__, "roomkit."):
        if any(part.startswith("_") for part in module_info.name.split(".")):
            continue
        try:
            module = importlib.import_module(module_info.name)
        except ImportError:
            continue
        for name, obj in vars(module).items():
            if (
                not name.startswith("_")
                and inspect.isclass(obj)
                and obj.__module__ == module.__name__
                and len(obj.__bases__) > 1
            ):
                found[f"{module.__name__}.{name}"] = module.__name__
    return found


def test_a_two_base_subclass_of_each_public_class_passes_mypy(tmp_path: Path) -> None:
    classes = _public_classes_with_several_bases()
    # The walk must reach the channels, the framework and the providers, or
    # this test would pass by checking nothing.
    for expected in (
        "roomkit.channels.acp.ACPChannel",
        "roomkit.core.framework.RoomKit",
        "roomkit.providers.gemini.realtime.GeminiLiveProvider",
    ):
        assert expected in classes
    checked = sorted(set(classes) - _BASES_THAT_DISAGREE)
    lines = ["from __future__ import annotations"]
    for index, qualified in enumerate(checked):
        name = qualified.rsplit(".", 1)[1]
        lines.append(f"from {classes[qualified]} import {name} as Base{index}")
    lines.append("class Mixin: ...")
    for index, qualified in enumerate(checked):
        name = qualified.rsplit(".", 1)[1]
        lines.append(f"class App{index}_{name}(Mixin, Base{index}): ...")
    application = tmp_path / "application.py"
    application.write_text("\n".join(lines) + "\n")

    # An empty --config-file reads no configuration; /dev/null as the cache
    # directory reads and writes no cache.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file=",
            "--no-incremental",
            "--cache-dir",
            os.devnull,
            str(application),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
