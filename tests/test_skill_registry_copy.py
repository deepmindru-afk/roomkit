"""A skill registry filled in memory, and copied whole or in part (RFC §24.3).

``add`` registers a skill built elsewhere (a store, a marketplace). ``copy``
keeps what the source finds: a skill discovered but not yet loaded is still
found, and the source's marks are kept unless the copy is asked not to.
"""

from __future__ import annotations

from pathlib import Path

from roomkit.skills import SkillRegistry
from tests.test_skills import _make_skill_dir_full


def _discovered(tmp_path: Path, *names: str) -> SkillRegistry:
    for name in names:
        _make_skill_dir_full(tmp_path, name, scripts=["run.py"])
    registry = SkillRegistry()
    registry.discover(tmp_path)
    return registry


def test_a_skill_built_in_memory_is_registered_and_found(tmp_path: Path) -> None:
    source = _discovered(tmp_path, "reports")
    skill = source.get_skill("reports")
    assert skill is not None
    registry = SkillRegistry()

    registry.add(skill)

    assert registry.skill_names == ["reports"]
    assert registry.get_skill("reports") is skill
    assert registry.get_skill("reports").resolve_script("run.py").name == "run.py"


def test_a_copy_finds_a_skill_discovered_but_not_loaded(tmp_path: Path) -> None:
    source = _discovered(tmp_path, "reports", "invoices")

    copied = source.copy(["reports"])

    assert copied.skill_names == ["reports"]
    found = copied.get_skill("reports")
    assert found is not None and found.name == "reports"
    assert copied.get_skill("invoices") is None


def test_a_copy_keeps_the_marks_unless_asked_not_to(tmp_path: Path) -> None:
    source = _discovered(tmp_path, "reports", "invoices", "legacy")
    source.mark_unlisted("invoices")
    source.mark_unavailable("legacy", "requires a tool not granted here")

    marked = source.copy()
    plain = source.copy(["reports", "invoices"], marks=False)

    assert marked.listed_names == ["reports"]
    assert marked.unavailable_skills == {"legacy": "requires a tool not granted here"}
    assert sorted(plain.listed_names) == ["invoices", "reports"]
    assert plain.unavailable_skills == {}


def test_adding_a_skill_clears_its_marks(tmp_path: Path) -> None:
    source = _discovered(tmp_path, "reports")
    skill = source.get_skill("reports")
    assert skill is not None
    source.mark_unavailable("reports", "not here")

    source.add(skill)

    assert source.unavailable_skills == {}
    assert source.skill_names == ["reports"]
