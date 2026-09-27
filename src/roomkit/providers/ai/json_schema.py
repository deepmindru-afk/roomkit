"""The portable JSON Schema subset a response schema must stay within.

Every provider that constrains its output accepts a different dialect of JSON
Schema: OpenAI's strict mode wants every property required and no extra keys,
Anthropic refuses numeric bounds, Gemini reads its own subset. What all of them
accept is the intersection checked here, so a schema that passes runs on any
provider that supports response schemas (RFC §6.7). It is deliberately narrow:
widening it later breaks no caller, narrowing it would.

The same subset is small enough to check an answer against without a JSON
Schema library: :func:`schema_mismatch` is how a provider makes sure the
document it returns satisfies the schema, whatever the server did with it.

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


def schema_mismatch(schema: Mapping[str, Any], value: Any, path: str = "$") -> str | None:
    """Where ``value`` departs from a portable ``schema``, or ``None`` when it fits.

    ``value`` is a parsed JSON document and ``schema`` one that passed
    :func:`check_portable_schema`. An object must carry exactly its properties,
    an ``integer`` is a whole number, and a boolean is never taken for a number.
    """
    kind = schema["type"]
    if kind == "object":
        return _object_mismatch(schema, value, path)
    if kind == "array":
        if not isinstance(value, list):
            return f"{path}: expected an array"
        for index, item in enumerate(value):
            found = schema_mismatch(schema["items"], item, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if not _SCALAR_CHECKS[kind](value):
        return f"{path}: expected {kind}, got {type(value).__name__}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path}: {value!r} is not one of {schema['enum']}"
    return None


def _object_mismatch(schema: Mapping[str, Any], value: Any, path: str) -> str | None:
    if not isinstance(value, dict):
        return f"{path}: expected an object"
    properties = schema["properties"]
    missing = [name for name in properties if name not in value]
    unexpected = [name for name in value if name not in properties]
    if missing or unexpected:
        return f"{path}: missing {missing}, unexpected {unexpected}"
    for name, subschema in properties.items():
        found = schema_mismatch(subschema, value[name], f"{path}.{name}")
        if found is not None:
            return found
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


_SCALAR_CHECKS = {
    "string": lambda value: isinstance(value, str),
    "boolean": lambda value: isinstance(value, bool),
    "number": _is_number,
    "integer": lambda value: (
        (isinstance(value, int) and not isinstance(value, bool))
        or (isinstance(value, float) and value.is_integer())
    ),
}


def _check(schema: Any, path: str) -> None:
    if not isinstance(schema, Mapping) or not all(isinstance(key, str) for key in schema):
        raise ValueError(f"{path}: a subschema must be an object with string keys")
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
    if not isinstance(properties, Mapping) or not all(isinstance(k, str) for k in properties):
        raise ValueError(f"{path}: an object must declare 'properties', keyed by name")
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
