"""Canonical prompt serialization for Yasama, Yurutme and Yargi.

The layout follows the training contracts documented in PROJECT_MASTER.md.  Keep
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


# How much of each observation a role sees (PROJECT_MASTER §3.2, §4).  Nothing
# is truncated:
# - Yasama only decides the next action, so it sees what ran and how it ended
#   (action / status / error), not file or page content.
# - Yurutme fills arguments from earlier results (e.g. FILE_EDIT after a
#   FILE_READ), so it sees observations in full.
# - Yargi answers from the content and sees observations in full.
# STATE carries the latest observation twice (last_observation and the end of
# recent_observations); it is written once, which halves the size of a long
# read without dropping anything.
_OBSERVATION_OUTCOME_FIELDS = ("action", "status", "error")


def _outcome(observation: Any) -> Any:
    if not isinstance(observation, Mapping):
        return observation
    return {key: observation.get(key) for key in _OBSERVATION_OUTCOME_FIELDS}


def _deduplicated_state(state: Mapping[str, Any]) -> dict[str, Any]:
    serialized = dict(state)
    recent = serialized.get("recent_observations")
    last = serialized.get("last_observation")
    if last is not None and isinstance(recent, Sequence) and recent and recent[-1] == last:
        del serialized["last_observation"]
    return serialized


def yasama_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """STATE as Yasama sees it: observation outcomes without their data."""
    serialized = _deduplicated_state(state)
    if serialized.get("last_observation") is not None:
        serialized["last_observation"] = _outcome(serialized["last_observation"])
    if isinstance(serialized.get("recent_observations"), Sequence):
        serialized["recent_observations"] = [
            _outcome(observation) for observation in serialized["recent_observations"]
        ]
    return serialized


def yurutme_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """STATE as Yurutme sees it: full observations, each written once."""
    return _deduplicated_state(state)


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
    # Once a tool has run in this turn the decision is whether the current
    # request is done; the step chain in STATE answers that, and earlier
    # conversation turns only invite copying their actions.
    if not state.get("completed_actions"):
        history = _history_block(conversation_history)
        if history is not None:
            parts.append(history)
    parts.append("STATE:\n" + _json(yasama_state(state)))
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
            "STATE:\n" + _json(yurutme_state(state)),
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
