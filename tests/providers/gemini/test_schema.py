"""Tests for Gemini JSON Schema cleaning."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.providers.gemini.schema import clean_gemini_schema


class TestCleanGeminiSchema:
    def test_strips_unsupported_fields(self) -> None:
        """Should remove $schema, additionalProperties, default, title."""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "SearchArgs",
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "title": "Query",
                    "default": "",
                },
            },
            "required": ["query"],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        assert "$schema" not in cleaned
        assert "title" not in cleaned
        assert "additionalProperties" not in cleaned
        assert cleaned["type"] == "object"
        assert cleaned["required"] == ["query"]
        # Nested property should also be cleaned
        assert "title" not in cleaned["properties"]["query"]
        assert "default" not in cleaned["properties"]["query"]
        assert cleaned["properties"]["query"]["type"] == "string"

    def test_preserves_valid_keys(self) -> None:
        """Valid Gemini schema keys should be preserved."""
        schema = {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of results",
                    "minimum": 1,
                    "maximum": 100,
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["count"],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned == schema  # unchanged — all keys are valid

    def test_nested_schemas(self) -> None:
        """Should recursively clean nested property schemas."""
        schema = {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "object",
                    "title": "Filter",
                    "additionalProperties": True,
                    "properties": {
                        "field": {
                            "type": "string",
                            "default": "name",
                            "title": "Field",
                        },
                    },
                },
            },
        }
        cleaned = clean_gemini_schema(schema)
        assert "title" not in cleaned["properties"]["filter"]
        assert "additionalProperties" not in cleaned["properties"]["filter"]
        nested = cleaned["properties"]["filter"]["properties"]["field"]
        assert "title" not in nested
        assert "default" not in nested
        assert nested["type"] == "string"

    def test_items_cleaned(self) -> None:
        """Array items schema should also be cleaned."""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "title": "Item",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "default": ""},
                },
            },
        }
        cleaned = clean_gemini_schema(schema)
        assert "title" not in cleaned["items"]
        assert "additionalProperties" not in cleaned["items"]

    def test_none_input(self) -> None:
        """None input should return None."""
        assert clean_gemini_schema(None) is None

    def test_empty_schema(self) -> None:
        """Empty dict should return empty dict."""
        assert clean_gemini_schema({}) == {}


class TestUnionCollapse:
    """Pydantic / OpenAPI union shapes collapse to a single nullable branch.

    Without this collapse, ``anyOf`` / ``oneOf`` / ``allOf`` would be
    stripped by the unknown-key pass and the property would emerge
    typeless — Gemini silently refuses to invoke such tools.
    """

    def test_pydantic_optional_to_nullable(self) -> None:
        """``Optional[str]`` from Pydantic becomes ``{type: string, nullable: True}``."""
        schema = {
            "type": "object",
            "properties": {
                "note": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Optional note",
                },
            },
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        note = cleaned["properties"]["note"]
        assert note["type"] == "string"
        assert note["nullable"] is True
        assert note["description"] == "Optional note"
        assert "anyOf" not in note

    def test_one_of_handled_like_any_of(self) -> None:
        schema = {
            "oneOf": [{"type": "integer"}, {"type": "null"}],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned == {"type": "integer", "nullable": True}

    def test_all_of_handled_like_any_of(self) -> None:
        schema = {
            "allOf": [{"type": "boolean"}, {"type": "null"}],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned == {"type": "boolean", "nullable": True}

    def test_wider_union_keeps_first_non_null(self) -> None:
        """Multiple non-null branches → keep first; nullable=True if any null."""
        schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"},
                {"type": "null"},
            ],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        assert cleaned["type"] == "string"
        assert cleaned["nullable"] is True

    def test_union_without_null_no_nullable(self) -> None:
        """Pure non-null union → first branch, no nullable flag."""
        schema = {
            "anyOf": [{"type": "string"}, {"type": "integer"}],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        assert cleaned["type"] == "string"
        assert "nullable" not in cleaned

    def test_pure_null_union_falls_back_to_string(self) -> None:
        """Degenerate {anyOf: [{type: null}]} stays valid."""
        schema = {"anyOf": [{"type": "null"}]}
        cleaned = clean_gemini_schema(schema)
        assert cleaned == {"type": "string", "nullable": True}

    def test_collapse_inside_array_items(self) -> None:
        """Pydantic Optional inside array items must also collapse."""
        schema = {
            "type": "array",
            "items": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
            },
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        items = cleaned["items"]
        assert items["type"] == "string"
        assert items["nullable"] is True

    def test_collapse_inside_nested_properties(self) -> None:
        """Pydantic Optional inside nested object properties must collapse."""
        schema = {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "description": "An optional int",
                        },
                    },
                },
            },
        }
        cleaned = clean_gemini_schema(schema)
        value = cleaned["properties"]["inner"]["properties"]["value"]  # type: ignore[index]
        assert value["type"] == "integer"
        assert value["nullable"] is True
        assert value["description"] == "An optional int"

    def test_description_preserved_when_collapsing(self) -> None:
        """Parent-level description must survive the collapse."""
        schema = {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "A field description.",
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        assert cleaned["description"] == "A field description."

    def test_branch_description_preserved(self) -> None:
        """A non-null branch's own description survives."""
        schema = {
            "anyOf": [
                {"type": "string", "description": "branch desc"},
                {"type": "null"},
            ],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        assert cleaned["description"] == "branch desc"

    def test_property_with_collapsed_optional_has_type(self) -> None:
        """Regression guard: properties never emerge typeless from a collapse."""
        schema = {
            "type": "object",
            "properties": {
                "field_a": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "field_b": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
            },
        }
        cleaned = clean_gemini_schema(schema)
        for prop in cleaned["properties"].values():  # type: ignore[union-attr]
            assert "type" in prop, f"Property emerged typeless: {prop}"


class TestTypeListCollapse:
    """JSON Schema's own spelling of optionality: ``type`` as a list.

    ``{"type": ["string", "null"]}`` is what the spec says and what a
    generator that is not Pydantic emits — a TypeScript MCP server through
    ``zod-to-json-schema``, for one. ``type`` is a key Gemini accepts, so such
    a list used to travel through the cleaning untouched and blow up inside
    ``FunctionDeclaration``, whose ``type`` is a single-valued enum.
    """

    def test_optional_string_collapses_to_nullable(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "cmd": {"type": ["string", "null"], "description": "the command"},
            },
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        cmd = cleaned["properties"]["cmd"]
        assert cmd["type"] == "string"
        assert cmd["nullable"] is True
        assert cmd["description"] == "the command"

    def test_list_without_null_keeps_first_and_stays_non_nullable(self) -> None:
        """Gemini has no union type, so a wider list keeps its first member —
        the same call :func:`_collapse_union` makes for a wider ``anyOf``."""
        cleaned = clean_gemini_schema({"type": ["string", "integer"]})
        assert cleaned == {"type": "string"}

    def test_wider_list_keeps_first_non_null(self) -> None:
        cleaned = clean_gemini_schema({"type": ["null", "integer", "string"]})
        assert cleaned == {"type": "integer", "nullable": True}

    def test_all_null_list_falls_back_to_string(self) -> None:
        """A typeless property is the failure this module exists to prevent."""
        cleaned = clean_gemini_schema({"type": ["null"]})
        assert cleaned == {"type": "string", "nullable": True}

    def test_single_element_list_is_still_a_list(self) -> None:
        cleaned = clean_gemini_schema({"type": ["boolean"]})
        assert cleaned == {"type": "boolean"}

    def test_scalar_type_is_left_alone(self) -> None:
        cleaned = clean_gemini_schema({"type": "string", "description": "d"})
        assert cleaned == {"type": "string", "description": "d"}

    def test_collapse_inside_array_items(self) -> None:
        schema = {"type": "array", "items": {"type": ["string", "null"]}}
        cleaned = clean_gemini_schema(schema)
        assert cleaned == {
            "type": "array",
            "items": {"type": "string", "nullable": True},
        }

    def test_collapse_in_nested_properties(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {"inner": {"type": ["integer", "null"]}},
                },
            },
        }
        cleaned = clean_gemini_schema(schema)
        inner = cleaned["properties"]["outer"]["properties"]["inner"]  # type: ignore[index]
        assert inner == {"type": "integer", "nullable": True}

    def test_type_list_inside_a_union_branch(self) -> None:
        """Both spellings at once: the union collapse picks a branch, and that
        branch's own type list still has to be folded."""
        schema = {"anyOf": [{"type": ["string", "null"]}, {"type": "null"}]}
        cleaned = clean_gemini_schema(schema)
        assert cleaned == {"type": "string", "nullable": True}

    def test_every_property_reaches_gemini_declarable(self) -> None:
        """The full flow, not just the dict: the cleaned schema has to be
        accepted by ``FunctionDeclaration`` itself, which is where a type list
        failed — a session whose tools carry one never connects at all."""
        from google.genai import types

        schema = {
            "type": "object",
            "properties": {
                "cmd": {"type": ["string", "null"], "description": "the command"},
                "count": {"type": ["integer", "null"]},
                "flag": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": ["string", "null"]}},
                "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
        }
        declaration = types.FunctionDeclaration(
            name="run",
            description="run something",
            parameters=clean_gemini_schema(schema),  # type: ignore[arg-type]
        )
        assert declaration.parameters is not None
        assert declaration.parameters.properties is not None
        assert set(declaration.parameters.properties) == {
            "cmd",
            "count",
            "flag",
            "tags",
            "note",
        }


def _assert_declarable(node: dict[str, Any], path: str = "parameters") -> None:
    """The shapes Gemini refuses, measured 2026-09-28 with a real call.

    ``FunctionDeclaration`` builds them without complaint, so the check has
    to read the cleaned dict: ``properties`` / ``required`` on a node that is
    not ``type: object`` ("only allowed for OBJECT type"), a ``required``
    name that ``properties`` does not define ("property is not defined"), and
    an array without ``items``, or ``items`` on a non-array.
    """
    if "properties" in node or "required" in node:
        assert node.get("type") == "object", f"{path}: properties on a non-object"
    if "items" in node or node.get("type") == "array":
        assert node.get("type") == "array", f"{path}: items on a non-array"
        assert isinstance(node.get("items"), dict), f"{path}: array without items"
    missing = set(node.get("required", [])) - set(node.get("properties", {}))
    assert not missing, f"{path}: required names undefined properties {missing}"
    for name, child in node.get("properties", {}).items():
        _assert_declarable(child, f"{path}.properties[{name}]")
    if isinstance(node.get("items"), dict):
        _assert_declarable(node["items"], f"{path}.items")


# A stored-file reference: an object whose ``oneOf`` only narrows it ("an
# upload needs its version"), the shape a file-accepting MCP tool declares.
_FILE_REFERENCE = {
    "type": "object",
    "description": "A stored file",
    "properties": {
        "source": {"type": "string", "enum": ["upload", "library"]},
        "id": {"type": "string", "format": "uuid"},
        "version": {"type": "integer", "minimum": 1},
    },
    "required": ["source", "id"],
    "additionalProperties": False,
    "oneOf": [
        {"properties": {"source": {"enum": ["upload"]}}, "required": ["version"]},
        {"properties": {"source": {"enum": ["library"]}}, "not": {"required": ["version"]}},
    ],
}


class TestRefiningUnion:
    """A union that narrows its node is not a choice between types.

    Folding one to its first branch, as ``Optional[X]`` is folded, would
    replace the object with an untyped fragment requiring a property it does
    not declare, and Gemini refuses the whole request over it (RMK-266).
    """

    def test_constraint_one_of_keeps_the_object(self) -> None:
        cleaned = clean_gemini_schema(_FILE_REFERENCE)
        assert cleaned == {
            "type": "object",
            "description": "A stored file",
            "properties": {
                "source": {"type": "string", "enum": ["upload", "library"]},
                "id": {"type": "string", "format": "uuid"},
                "version": {"type": "integer", "minimum": 1},
            },
            "required": ["source", "id"],
        }

    def test_either_or_root_keeps_every_property(self) -> None:
        """The "give url or path" tool keeps its type and all its properties,
        where the fold would leave ``{"required": ["url"]}``."""
        schema = {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "path": {"type": "string"},
                "pages": {"type": "integer"},
            },
            "oneOf": [{"required": ["url"]}, {"required": ["path"]}],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned == {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "path": {"type": "string"},
                "pages": {"type": "integer"},
            },
        }

    def test_fields_a_branch_adds_are_declared_without_its_required(self) -> None:
        """``url`` or ``path`` beside ``mode``: both are offered, neither is
        required, since only one branch applies."""
        schema = {
            "type": "object",
            "properties": {"mode": {"type": "string"}},
            "required": ["mode"],
            "oneOf": [
                {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
                {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned == {
            "type": "object",
            "properties": {
                "mode": {"type": "string"},
                "url": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["mode"],
        }

    def test_all_of_mixin_adds_its_fields_and_the_node_wins(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string", "description": "own"}},
            "allOf": [{"properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}}}],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned == {
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "own"},
                "b": {"type": "boolean"},
            },
        }

    def test_array_keeps_its_items_under_a_typed_union(self) -> None:
        """The node's own ``items`` is its shape: folding to the first branch
        would drop it, and Gemini refuses an array without ``items``. The null
        branch still makes it nullable, as the fold would."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "anyOf": [{"type": "array", "minItems": 1}, {"type": "null"}],
        }
        assert clean_gemini_schema(schema) == {
            "type": "array",
            "items": {"type": "string"},
            "nullable": True,
        }

    def test_a_null_branch_beside_untyped_ones_keeps_the_scalar_nullable(self) -> None:
        schema = {"type": "string", "anyOf": [{"format": "date"}, {"type": "null"}]}
        assert clean_gemini_schema(schema) == {"type": "string", "nullable": True}

    def test_a_nullable_object_type_list_still_gains_the_union_fields(self) -> None:
        schema = {
            "type": ["object", "null"],
            "properties": {"a": {"type": "string"}},
            "oneOf": [{"properties": {"b": {"type": "string"}}, "required": ["b"]}],
        }
        assert clean_gemini_schema(schema) == {
            "type": "object",
            "nullable": True,
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        }

    def test_all_of_intersection_gathers_every_branch(self) -> None:
        """Every branch of an ``allOf`` applies: their fields and their
        ``required`` all hold, where an ``anyOf`` offers one of them."""
        schema = {
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"type": "object", "properties": {"b": {"type": "integer"}}, "required": ["b"]},
            ],
        }
        assert clean_gemini_schema(schema) == {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }

    def test_optional_discriminated_union_folds_to_its_first_member(self) -> None:
        """Pydantic's ``Optional[Annotated[Cat | Dog, Field(discriminator=...)]]``:
        an ``anyOf`` whose first branch is a ``oneOf``. Folding the outer
        union alone left ``{"nullable": true}``, with no type."""
        schema = {
            "description": "The pet",
            "anyOf": [
                {
                    "oneOf": [
                        {"type": "object", "properties": {"meows": {"type": "boolean"}}},
                        {"type": "object", "properties": {"barks": {"type": "boolean"}}},
                    ],
                    "discriminator": {"propertyName": "kind"},
                },
                {"type": "null"},
            ],
        }
        assert clean_gemini_schema(schema) == {
            "type": "object",
            "properties": {"meows": {"type": "boolean"}},
            "nullable": True,
            "description": "The pet",
        }

    def test_untyped_all_of_mixin_becomes_an_object(self) -> None:
        schema = {
            "description": "Filters",
            "allOf": [
                {"properties": {"a": {"type": "string"}}},
                {"properties": {"b": {"type": "integer"}}},
            ],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned == {
            "description": "Filters",
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        }

    def test_untyped_branches_keep_a_scalar(self) -> None:
        schema = {"type": "string", "anyOf": [{"format": "date"}, {"format": "date-time"}]}
        assert clean_gemini_schema(schema) == {"type": "string"}

    def test_empty_union_takes_the_typed_fallback(self) -> None:
        """An empty union narrows nothing, and a typeless node is what this
        module exists to prevent."""
        assert clean_gemini_schema({"anyOf": []}) == {"type": "string", "nullable": True}

    def test_typed_union_on_a_bare_object_still_folds(self) -> None:
        """``{"type": "object"}`` without properties of its own offers its
        branches as the alternatives: the first one is still the shape."""
        schema = {
            "type": "object",
            "anyOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "object", "properties": {"b": {"type": "integer"}}},
            ],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned == {"type": "object", "properties": {"a": {"type": "string"}}}

    def test_nested_tool_is_declarable(self) -> None:
        """An MCP tool that nests file references in an array, next to
        free-form objects, which Gemini accepts and are kept."""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "attachments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "labels": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                            "metadata": {
                                "anyOf": [
                                    {"type": "object", "additionalProperties": True},
                                    {"type": "null"},
                                ],
                            },
                            "document_ref": _FILE_REFERENCE,
                            "preview_ref": _FILE_REFERENCE,
                        },
                        "required": ["name"],
                    },
                },
            },
            "required": ["title", "attachments"],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        _assert_declarable(cleaned)
        item = cleaned["properties"]["attachments"]["items"]["properties"]
        assert item["labels"] == {"type": "object"}
        assert item["metadata"] == {"type": "object", "nullable": True}
        assert set(item["preview_ref"]["properties"]) == {"source", "id", "version"}

    def test_empty_root_is_left_alone(self) -> None:
        schema = {"type": "object", "properties": {}}
        assert clean_gemini_schema(schema) == schema


class TestImpliedShape:
    """What JSON Schema leaves implied and Gemini refuses unless spelled out,
    each a 400 for the whole request (measured 2026-09-28)."""

    def test_properties_without_type_make_an_object(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "opts": {"properties": {"a": {"type": "string"}}, "required": ["a"]},
            },
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        assert cleaned["properties"]["opts"] == {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        }
        _assert_declarable(cleaned)

    def test_untyped_either_or_object_keeps_its_properties(self) -> None:
        schema = {
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "oneOf": [{"required": ["a"]}, {"required": ["b"]}],
        }
        assert clean_gemini_schema(schema) == {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        }

    def test_items_without_type_make_an_array(self) -> None:
        cleaned = clean_gemini_schema({"items": {"type": "string"}})
        assert cleaned == {"type": "array", "items": {"type": "string"}}

    def test_array_without_items_takes_any_value(self) -> None:
        """``items: {}`` is what Pydantic sends for ``list[Any]``."""
        assert clean_gemini_schema({"type": "array"}) == {"type": "array", "items": {}}

    def test_tuple_items_become_any_value(self) -> None:
        schema = {"type": "array", "items": [{"type": "string"}, {"type": "integer"}]}
        assert clean_gemini_schema(schema) == {"type": "array", "items": {}}

    def test_optional_array_without_items_after_the_fold(self) -> None:
        cleaned = clean_gemini_schema({"anyOf": [{"type": "array"}, {"type": "null"}]})
        assert cleaned == {"type": "array", "nullable": True, "items": {}}

    def test_keys_the_type_cannot_carry_are_dropped(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "both": {"properties": {"a": {"type": "string"}}, "items": {"type": "string"}},
                "listed": {"type": ["string", "array"], "items": {"type": "string"}},
                "array_under_union": {
                    "items": {"type": "string"},
                    "anyOf": [{"properties": {"a": {"type": "string"}}}],
                },
                "string_with_properties": {
                    "type": "string",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                },
            },
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        _assert_declarable(cleaned)
        props = cleaned["properties"]
        assert props["both"] == {"type": "object", "properties": {"a": {"type": "string"}}}
        assert props["listed"] == {"type": "string"}
        assert props["array_under_union"] == {"type": "array", "items": {"type": "string"}}
        assert props["string_with_properties"] == {"type": "string"}


class TestRequiredMatchesProperties:
    def test_required_drops_a_property_the_cleaning_dropped(self) -> None:
        """A boolean schema (``"x": true``, "anything") is not a dict and does
        not survive; naming it in ``required`` would fail the request."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "x": True},
            "required": ["a", "x"],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned == {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        }

    def test_required_left_empty_is_dropped(self) -> None:
        assert clean_gemini_schema({"type": "object", "required": ["a"]}) == {"type": "object"}
        assert clean_gemini_schema({"type": "string", "required": ["a"]}) == {"type": "string"}

    def test_a_name_that_is_not_a_string_is_dropped(self) -> None:
        """One malformed tool must not raise while every tool of the request
        is being cleaned."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": [["a"], "a"],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        assert cleaned["required"] == ["a"]


class TestNonStringEnum:
    """Gemini's ``enum`` holds strings: other values go to the description (RMK-281)."""

    @pytest.mark.parametrize(
        ("prop", "expected"),
        [
            (
                {"type": "integer", "enum": [1, 2, 3], "description": "Urgency."},
                {"type": "integer", "description": "Urgency. Allowed values: 1, 2, 3."},
            ),
            (
                {"type": "number", "enum": [0.5, 1.5]},
                {"type": "number", "description": "Allowed values: 0.5, 1.5."},
            ),
            (
                {"type": "boolean", "enum": [True]},
                {"type": "boolean", "description": "Allowed values: true."},
            ),
            (
                {"enum": ["a", 1, None]},
                {"description": 'Allowed values: "a", 1, null.'},
            ),
        ],
    )
    def test_the_values_move_to_the_description_and_the_type_stays(
        self, prop: dict[str, Any], expected: dict[str, Any]
    ) -> None:
        schema = {"type": "object", "properties": {"p": prop}, "required": ["p"]}

        cleaned = clean_gemini_schema(schema)

        assert cleaned == {"type": "object", "properties": {"p": expected}, "required": ["p"]}

    def test_a_string_enum_is_kept(self) -> None:
        prop = {"type": "string", "enum": ["low", "high"]}

        cleaned = clean_gemini_schema({"type": "object", "properties": {"p": prop}})

        assert cleaned == {"type": "object", "properties": {"p": prop}}

    def test_a_nested_enum_is_described_too(self) -> None:
        schema = {
            "type": "object",
            "properties": {"levels": {"type": "array", "items": {"enum": [1, 2]}}},
        }

        cleaned = clean_gemini_schema(schema)

        assert cleaned["properties"]["levels"]["items"] == {"description": "Allowed values: 1, 2."}

    def test_a_tuple_of_items_is_declarable(self) -> None:
        schema = {
            "type": "object",
            "properties": {"point": {"type": "array", "items": [{"type": "number"}] * 2}},
        }

        cleaned = clean_gemini_schema(schema)

        assert cleaned["properties"]["point"] == {"type": "array", "items": {}}
