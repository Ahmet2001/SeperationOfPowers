"""Yasama runtime: chooses the next canonical action only."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


CANONICAL_ACTIONS = frozenset(
    {
        "RESPOND",
        "WEB_SEARCH",
        "WEB_OPEN",
        "FILE_READ",
        "FILE_SEARCH",
        "FILE_LIST",
        "FILE_CREATE",
        "FILE_EDIT",
        "FILE_DELETE",
        "FILE_MOVE",
        "CODE_EXECUTE",
        "SHELL_EXECUTE",
        "DOWNLOAD",
        "SEND_MAIL",
        "ASK_CLARIFICATION",
        "FINISH",
    }
)

InferenceFn = Callable[[dict[str, Any]], str]


class YasamaRuntime:
    """Owns Yasama inference and validates its single-action output."""

    def __init__(self, inference_fn: InferenceFn | None = None) -> None:
        self._inference_fn = inference_fn

    def run(
        self,
        *,
        user_prompt: str,
        conversation_history: Sequence[Mapping[str, Any]],
        state: Mapping[str, Any],
    ) -> str:
        payload = {
            "user_prompt": user_prompt,
            "conversation_history": list(conversation_history),
            "state": dict(state),
        }

        action = self._infer(payload).strip().upper()
        if action not in CANONICAL_ACTIONS:
            raise ValueError(f"Invalid Yasama action: {action!r}")

        return action

    def _infer(self, payload: dict[str, Any]) -> str:
        if self._inference_fn is None:
            raise RuntimeError(
                "Yasama model runtime is not configured. "
                "Pass an inference_fn or implement model loading in yasama/yasama.py."
            )
        return self._inference_fn(payload)
