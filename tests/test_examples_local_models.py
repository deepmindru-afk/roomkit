"""Every local model an example names exists, so its download command works."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from roomkit.local_models import LOCAL_MODELS

EXAMPLES = Path(__file__).parents[1] / "examples"
KNOWN = {model.name for model in LOCAL_MODELS}
# A download command in a docstring, its names possibly continued on the next lines.
DOWNLOAD = re.compile(
    r"python -m roomkit\.local_models download((?:[ \t]+[\w.\-]+|[ \t]*\\\\?\n)+)"
)


def _named_in_calls(tree: ast.AST) -> set[str]:
    """String literals passed to from_local_model() and require_local_models()."""
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if called == "from_local_model" and node.args:
            candidates = node.args[:1]
        elif called == "require_local_models":
            candidates = node.args
        else:
            continue
        names |= {
            arg.value
            for arg in candidates
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        }
    return names


def _named_in_download_commands(source: str) -> set[str]:
    names = set()
    for match in DOWNLOAD.finditer(source):
        names |= {word for word in re.split(r"[\s\\]+", match.group(1)) if word}
    return names


EXAMPLES_USING_LOCAL_MODELS = sorted(
    path for path in EXAMPLES.rglob("*.py") if "local_model" in path.read_text()
)


def test_examples_use_local_models() -> None:
    assert len(EXAMPLES_USING_LOCAL_MODELS) >= 15


@pytest.mark.parametrize("path", EXAMPLES_USING_LOCAL_MODELS, ids=lambda p: p.name)
def test_every_named_model_exists(path: Path) -> None:
    source = path.read_text()
    named = _named_in_calls(ast.parse(source)) | _named_in_download_commands(source)
    assert named <= KNOWN, f"unknown local models: {sorted(named - KNOWN)}"
