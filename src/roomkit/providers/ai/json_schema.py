"""The portable JSON Schema subset a response schema must stay within.

Every provider that constrains its output accepts a different dialect of JSON
Schema: OpenAI's strict mode wants every property required and no extra keys,
Anthropic refuses numeric bounds, Gemini reads its own subset. What all of them
accept is the intersection checked here, so a schema that passes runs on any
provider that supports response schemas (RFC §6.7). It is deliberately narrow:
widening it later breaks no caller, narrowing it would.

Pure and dependency-free, so ``AIContext`` can validate the field without an
import cycle through the provider modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ANNOTATIONS = frozenset({"title", "description"})
_KEYWORDS_BY_TYPE: dict[str, frozenset[str]] = {
    "object": frozenset({"type", "properties", "required", "additionalProperties"}),
    "array": frozenset({"type", "items"}),
    "string": frozenset({"type", "enum"}),
    "number": frozenset({"type"}),
    "integer": frozenset({"type"}),
    "boolean": frozenset({"type"}),
}


def check_portable_schema(schema: Mapping[str, Any]) -> None:
    """Refuse a response schema outside the portable subset.

    The root must be an object. Every object lists ``properties``, names each
    of them in ``required`` and sets ``additionalProperties`` to ``false``;
    every array has ``items``; only a string may carry ``enum``. ``title`` and
    ``description`` are allowed anywhere.

    Raises:
        ValueError: naming the path of the first subschema that is not portable.
    """
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise ValueError("a response schema must be a JSON Schema whose root type is 'object'")
    _check(schema, "$")


def _check(schema: Any, path: str) -> None:
    if not isinstance(schema, Mapping):
        raise ValueError(f"{path}: a subschema must be an object")
    kind = schema.get("type")
    if not isinstance(kind, str) or kind not in _KEYWORDS_BY_TYPE:
        allowed = ", ".join(_KEYWORDS_BY_TYPE)
        raise ValueError(f"{path}: 'type' must be one of {allowed}, as a single string")
    extra = set(schema) - _KEYWORDS_BY_TYPE[kind] - _ANNOTATIONS
    if extra:
        raise ValueError(f"{path}: {sorted(extra)} not portable on a {kind!r} schema")
    for key in _ANNOTATIONS & set(schema):
        if not isinstance(schema[key], str):
            raise ValueError(f"{path}: {key!r} must be a string")
    if kind == "object":
        _check_object(schema, path)
    elif kind == "array":
        if "items" not in schema:
            raise ValueError(f"{path}: an array must declare 'items'")
        _check(schema["items"], f"{path}[]")
    elif kind == "string" and "enum" in schema:
        _check_enum(schema["enum"], path)


def _check_object(schema: Mapping[str, Any], path: str) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError(f"{path}: an object must declare 'properties'")
    if schema.get("additionalProperties") is not False:
        raise ValueError(f"{path}: an object must set 'additionalProperties' to false")
    required = schema.get("required")
    if (
        not isinstance(required, list)
        or not all(isinstance(name, str) for name in required)
        or sorted(required) != sorted(properties)
    ):
        raise ValueError(f"{path}: 'required' must list every property, each once")
    for name, subschema in properties.items():
        _check(subschema, f"{path}.{name}")


def _check_enum(values: Any, path: str) -> None:
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(v, str) for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{path}: 'enum' must be a non-empty list of distinct strings")
