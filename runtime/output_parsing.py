"""Compatibility parsers for model outputs.

Fine-tuned role models should follow the exact contracts. These helpers only make
runtime testing with generic instruct models less brittle when they add wrappers
such as ``ACTION: ...``, fenced JSON, or ``{"response": ...}``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any


_ACTION_LABEL = re.compile(
    r"^\s*(?:ACTION|EYLEM|NEXT_ACTION|TARGET)\s*:\s*([A-Z_]+)\b",
    re.IGNORECASE,
)


def parse_canonical_action(raw: str, actions: Iterable[str]) -> str:
    """Return one canonical action from an exact or lightly wrapped response."""

    allowed = {str(action).upper() for action in actions}
    text = raw.strip()
    upper = text.upper()
    if upper in allowed:
        return upper

    # Some generic instruct models return {"action": "FILE_READ"}.
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, Mapping):
        value = decoded.get("action")
        if isinstance(value, str) and value.strip().upper() in allowed:
            return value.strip().upper()

    # Accept a canonical action as the first non-empty line.
    for line in text.splitlines():
        stripped = line.strip().strip("`*")
        if not stripped:
            continue
        candidate = stripped.upper()
        if candidate in allowed:
            return candidate
        match = _ACTION_LABEL.match(stripped)
        if match:
            candidate = match.group(1).upper()
            if candidate in allowed:
                return candidate
        break

    raise ValueError(f"Invalid Yasama action: {upper!r}")


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating markdown fences or leading explanation."""

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, Mapping):
        return dict(decoded)

    start = text.find("{")
    if start >= 0:
        try:
            decoded, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, Mapping):
            return dict(decoded)

    raise ValueError("Yurutme produced invalid JSON object output.")


def unwrap_natural_language_response(raw: str) -> str:
    """Unwrap a generic-model ``{"response": "..."}`` result when present."""

    text = raw.strip()
    if not text:
        return text
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(decoded, Mapping) and isinstance(decoded.get("response"), str):
        return decoded["response"].strip()
    return text
