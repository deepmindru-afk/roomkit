"""Documents and examples import roomkit names from public paths.

A name is public where the ``__all__`` of ``roomkit`` or of one of its
packages lists it, imported from that package (README, "Public API"). What a
reader copies from an example or a page is what an application imports: a
module path there makes an internal module part of the API, and a name that
does not resolve fails whoever copies it. The docstrings under
``src/roomkit`` are read too: the API reference renders them. ``tests/
public_imports.py`` reads the imports; ``make docs`` runs it on the docs
site's pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.public_imports import check

_ROOT = Path(__file__).resolve().parents[1]
_DOCUMENTS = ("examples", "README.md", "AGENTS.md", "llms.txt", "docs/c7")


def test_documents_and_examples_import_from_public_paths() -> None:
    found = check([_ROOT / path for path in _DOCUMENTS], root=_ROOT)

    assert [str(finding) for finding in found] == []


def test_docstrings_import_from_public_paths() -> None:
    found = check([_ROOT / "src" / "roomkit"], root=_ROOT, docstrings=True)

    assert [str(finding) for finding in found] == []


def test_a_docstring_is_read_and_the_code_around_it_is_not(tmp_path: Path) -> None:
    """roomkit's own code imports its modules; only what it documents is read."""
    module = tmp_path / "module.py"
    module.write_text(
        "from roomkit.models.event import TextContent\n\n\n"
        "def build():\n"
        '    """Build one::\n\n'
        "        >>> from roomkit.models.event import RoomEvent\n"
        '    """\n',
        encoding="utf-8",
    )

    found = check([module], docstrings=True)

    assert [(f.line, f.statement) for f in found] == [
        (7, "from roomkit.models.event import RoomEvent")
    ]


@pytest.mark.parametrize(
    ("name", "text", "problem"),
    [
        ("page.md", "```python\nfrom roomkit import RoomKit, TextContent\n```\n", None),
        ("page.md", "```python\nfrom roomkit.voice.stt import DeepgramSTTProvider\n```\n", None),
        ("page.md", "```python\nimport roomkit.voice.stt\n```\n", None),
        ("page.md", "```python\nfrom roomkit.voice import stt\n```\n", None),
        ("page.md", "```python\nfrom roomkit import RoomKit\nkit = RoomKit(...)\n```\n", None),
        (
            "page.md",
            "```python\nfrom roomkit.models.event import TextContent\n```\n",
            "not public at roomkit.models.event; public as roomkit.TextContent,",
        ),
        (
            "page.md",
            "!!! note\n    ```python\n    from roomkit.models.event import TextContent\n    ```\n",
            "not public at roomkit.models.event",
        ),
        (
            "page.md",
            "Import it with `from roomkit.models.event import TextContent`.\n",
            "not public at roomkit.models.event",
        ),
        (
            "page.md",
            "```python\nimport roomkit.voice.stt.deepgram\n```\n",
            "a module path is internal",
        ),
        (
            "page.md",
            "```python\nfrom roomkit.video.vision.screen_input import _get_scale_factor\n```\n",
            "an underscore name is internal",
        ),
        (
            "page.md",
            "```python\nfrom roomkit import RetryPolicy\n```\n",
            "does not resolve: roomkit has no RetryPolicy; public as roomkit.models.RetryPolicy",
        ),
        (
            "example.py",
            '"""Swap one line::\n\n    from roomkit.models.event import TextContent\n"""\n',
            "not public at roomkit.models.event",
        ),
        ("page.md", "```python\nfrom roomkit import __version__\n```\n", None),
        (
            "example.py",
            "from roomkit.models.event import \\\n    TextContent\n",
            "not public at roomkit.models.event",
        ),
        (
            "example.py",
            "from roomkit.models.event import (  # one of them)\n    TextContent,\n)\n",
            "not public at roomkit.models.event",
        ),
    ],
    ids=[
        "public names",
        "a provider its package exports lazily",
        "a package imported",
        "a package imported from its parent",
        "a block that is not valid Python",
        "a module path",
        "a block indented in an admonition",
        "an inline code span",
        "a module import",
        "an underscore name",
        "a name that does not resolve",
        "a docstring in an example",
        "a dunder its package lists",
        "a backslash continuation",
        "a parenthesis in a comment",
    ],
)
def test_an_import_is_public_only_from_a_package_that_lists_it(
    tmp_path: Path, name: str, text: str, problem: str | None
) -> None:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")

    found = check([path])

    if problem is None:
        assert found == []
    else:
        assert len(found) == 1 and problem in found[0].problem, found


def test_a_name_whose_optional_dependency_fails_to_import_is_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fish Audio's module raises a plain ImportError of its own when msgpack
    is missing: the check leaves the name out, as for a missing module."""
    monkeypatch.setitem(sys.modules, "msgpack", None)
    for module in ("roomkit.voice.tts.fish_audio", "roomkit.voice.tts._fish_audio_ws"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    page = tmp_path / "page.md"
    page.write_text("```python\nfrom roomkit.voice.tts import FishAudioTTSProvider\n```\n")

    assert check([page]) == []
