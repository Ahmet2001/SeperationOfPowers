"""Canonical prompt serialization for Yasama, Yurutme and Yargi.

The layout follows the training contracts documented in MasterFormat.txt.  Keep
this module as the single runtime source of truth so backend changes do not
silently change what the models see.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def _json(value: Any) -> str:
    """Stable, human-readable JSON used by the training examples."""
    return json.dumps(value, ensure_ascii=False, indent=2)


def _history_block(
    conversation_history: Sequence[Mapping[str, Any]],
) -> str | None:
    if not conversation_history:
        return None
    return "CONVERSATION_HISTORY:\n" + _json(
        [dict(message) for message in conversation_history]
    )


def serialize_yasama_input(
    *,
    user_prompt: str,
    conversation_history: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> str:
    """Serialize Yasama input as USER + optional history + STATE.

    The model target remains a raw canonical action string and is intentionally
    not included here.
    """
    parts = [f"USER:\n{user_prompt}"]
    history = _history_block(conversation_history)
    if history is not None:
        parts.append(history)
    parts.append("STATE:\n" + _json(dict(state)))
    return "\n".join(parts)


def serialize_yurutme_input(
    *,
    user_prompt: str,
    conversation_history: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
    action: str,
    tool_schema: Mapping[str, Any],
    placeholder: Mapping[str, Any] | None,
) -> str:
    """Serialize Yurutme input without leaking the target arguments.

    Conceptual training input includes user/history/state/action/tool schema and
    placeholder.  The target is the arguments JSON only.
    """
    parts = [f"USER:\n{user_prompt}"]
    history = _history_block(conversation_history)
    if history is not None:
        parts.append(history)
    parts.extend(
        [
            "STATE:\n" + _json(dict(state)),
            f"ACTION:\n{action}",
            "TOOL_SCHEMA:\n" + _json(dict(tool_schema)),
            "PLACEHOLDER:\n"
            + _json(dict(placeholder) if placeholder is not None else None),
        ]
    )
    return "\n".join(parts)


def serialize_yargi_input(
    *,
    user_prompt: str,
    conversation_history: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> str:
    """Serialize Yargi input using the documented tool_observations block.

    With an empty history this intentionally matches the MasterFormat ChatML
    example: raw user request followed by <tool_observations> JSON.
    """
    parts = [user_prompt]
    history = _history_block(conversation_history)
    if history is not None:
        parts.append(history)
    parts.append(
        "<tool_observations>\n"
        + _json([dict(observation) for observation in observations])
        + "\n</tool_observations>"
    )
    return "\n".join(parts)
