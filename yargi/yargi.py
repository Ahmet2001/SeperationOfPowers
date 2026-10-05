"""Yargi runtime: composes a grounded user-facing response from real observations."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from runtime.backends import ChatBackend, GenerationConfig


YARGI_SYSTEM_PROMPT = """Sen Mercan Yargi modelisin.
Kullanicinin istegini ve yalnizca gercek executor observation'larini kullanarak nihai cevabi uret.
Observation icinde bulunmayan bir basari, sonuc veya bilgi uydurma.
Bir observation error durumundaysa islemi basarili gibi anlatma.
Ciktin kullaniciya gidecek dogal dil cevabi olmalidir; JSON wrapper kullanma.
"""

InferenceFn = Callable[[dict[str, Any]], str]


class YargiRuntime:
    """Owns Yargi inference and returns the final natural-language response."""

    def __init__(
        self,
        inference_fn: InferenceFn | None = None,
        *,
        backend: ChatBackend | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        if inference_fn is not None and backend is not None:
            raise ValueError("Configure either inference_fn or backend, not both.")
        self._inference_fn = inference_fn
        self._backend = backend
        self._generation_config = generation_config or GenerationConfig(
            max_tokens=768,
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.05,
        )

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
        if self._inference_fn is not None:
            return self._inference_fn(payload)

        if self._backend is not None:
            messages = [
                {"role": "system", "content": YARGI_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                },
            ]
            return self._backend.generate(messages, config=self._generation_config)

        raise RuntimeError(
            "Yargi model runtime is not configured. "
            "Pass an inference_fn or a ChatBackend."
        )
