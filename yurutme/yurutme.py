"""Yurutme runtime: fills arguments for the action selected by Yasama."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any


InferenceFn = Callable[[dict[str, Any]], Mapping[str, Any] | str]


class YurutmeRuntime:
    """Owns Yurutme inference and returns a JSON-compatible arguments object."""

    def __init__(self, inference_fn: InferenceFn | None = None) -> None:
        self._inference_fn = inference_fn

    def run(
        self,
        *,
        user_prompt: str,
        conversation_history: Sequence[Mapping[str, Any]],
        state: Mapping[str, Any],
        action: str,
        tool_schema: Mapping[str, Any],
        placeholder: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "user_prompt": user_prompt,
            "conversation_history": list(conversation_history),
            "state": dict(state),
            "action": action,
            "tool_schema": dict(tool_schema),
            "placeholder": dict(placeholder) if placeholder is not None else None,
        }

        raw_arguments = self._infer(payload)
        if isinstance(raw_arguments, str):
            try:
                raw_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("Yurutme produced invalid JSON.") from exc

        if not isinstance(raw_arguments, Mapping):
            raise TypeError("Yurutme output must be a JSON object.")

        return dict(raw_arguments)

    def _infer(self, payload: dict[str, Any]) -> Mapping[str, Any] | str:
        if self._inference_fn is None:
            raise RuntimeError(
                "Yurutme model runtime is not configured. "
                "Pass an inference_fn or implement model loading in yurutme/yurutme.py."
            )
        return self._inference_fn(payload)
