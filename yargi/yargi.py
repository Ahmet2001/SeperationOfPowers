"""Yargi runtime: composes a grounded user-facing response from real observations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


InferenceFn = Callable[[dict[str, Any]], str]


class YargiRuntime:
    """Owns Yargi inference and returns the final natural-language response."""

    def __init__(self, inference_fn: InferenceFn | None = None) -> None:
        self._inference_fn = inference_fn

    def run(
        self,
        *,
        user_prompt: str,
        conversation_history: Sequence[Mapping[str, Any]],
        observations: Sequence[Mapping[str, Any]],
    ) -> str:
        payload = {
            "user_prompt": user_prompt,
            "conversation_history": list(conversation_history),
            "observations": [dict(observation) for observation in observations],
        }

        response = self._infer(payload).strip()
        if not response:
            raise ValueError("Yargi produced an empty response.")

        return response

    def _infer(self, payload: dict[str, Any]) -> str:
        if self._inference_fn is None:
            raise RuntimeError(
                "Yargi model runtime is not configured. "
                "Pass an inference_fn or implement model loading in yargi/yargi.py."
            )
        return self._inference_fn(payload)
