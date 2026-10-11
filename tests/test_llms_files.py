"""The llms files ship in the package and are copied to the docs site.

llms-full.txt is built from docs/c7 (``scripts/build_llms_full.py``), and both
llms files name the release they describe. A hand edit of llms-full.txt, or a
release that leaves the version line behind, fails here rather than reaching
an assistant that reads them.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "build_llms_full.py"
_VERSION_LINE = re.compile(r"^> Minimum version: (\S+) \|", re.MULTILINE)
_RELEASE_HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?)\]", re.MULTILINE)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_llms_full", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _latest_release() -> str:
    match = _RELEASE_HEADING.search((_ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    assert match, "CHANGELOG.md has no released version heading"
    return match.group(1)


def test_llms_full_is_built_from_the_topic_pages() -> None:
    committed = (_ROOT / "llms-full.txt").read_text(encoding="utf-8")
    assert committed == _load_script().assemble(_ROOT), (
        "llms-full.txt differs from docs/c7: edit the topic pages, then run `make llms-full`"
    )


def test_llms_files_name_the_latest_release() -> None:
    latest = _latest_release()
    for name in ("llms.txt", "docs/c7/overview.md"):
        match = _VERSION_LINE.search((_ROOT / name).read_text(encoding="utf-8"))
        assert match, f"{name} has no '> Minimum version:' line"
        assert match.group(1) == latest, (
            f"{name} names {match.group(1)}, CHANGELOG.md's latest release is {latest}"
        )
