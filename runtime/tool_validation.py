"""Small dependency-free validator for role-produced tool arguments.

This is a runtime contract check, not a complete JSON Schema implementation.
It validates the subset used by built-in tool definitions: object, required,
properties, simple primitive types and additionalProperties=False.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    # Unsupported schema types must not silently pass.
    return False


def validate_tool_arguments(
    action: str, arguments: Any, tool_schema: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject non-argument wrappers and missing/invalid fields before execution."""
    if not isinstance(arguments, Mapping):
        raise ValueError(f"{action} arguments must be a JSON object.")

    params = tool_schema.get("parameters")
    if not isinstance(params, Mapping):
        return dict(arguments)

    if params.get("type") == "object":
        properties = params.get("properties", {})
        required = params.get("required", [])
        no_extras = params.get("additionalProperties") is False
    else:
        # Legacy tests/configs sometimes provide a flat parameter map.
        properties = params
        required = []
        no_extras = False

    if not isinstance(properties, Mapping):
        raise ValueError(f"{action} tool schema has invalid properties.")
    if not isinstance(required, list) or any(not isinstance(k, str) for k in required):
        raise ValueError(f"{action} tool schema has invalid required fields.")

    unknown = set(arguments) - set(properties)
    if unknown and no_extras:
        raise ValueError(
            f"{action} returned unexpected argument fields: {', '.join(sorted(map(str, unknown)))}."
        )

    missing = [key for key in required if key not in arguments]
    if missing:
        raise ValueError(f"{action} is missing required arguments: {', '.join(missing)}.")

    for key, value in arguments.items():
        field = properties.get(key)
        if not isinstance(field, Mapping):
            continue
        expected = field.get("type")
        if not isinstance(expected, str):
            continue
        if not _matches_type(value, expected):
            raise ValueError(
                f"{action}.{key} must have type {expected}, got {type(value).__name__}."
            )
        if key in required and expected == "string" and not value.strip():
            raise ValueError(f"{action}.{key} cannot be empty.")

    return dict(arguments)
