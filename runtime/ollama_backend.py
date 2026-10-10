"""Ollama backend tuned for deterministic role runtimes.

Qwen3-class models can spend the entire small generation budget in Ollama's
separate ``message.thinking`` field and leave ``message.content`` empty.  The
Separation-of-Powers roles need the actual contract output (action / JSON /
final answer), so thinking is disabled by default for this backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .backends import ChatMessage, GenerationConfig, _normalized_messages, _post_json


class OllamaBackend:
    """Ollama ``/api/chat`` backend.

    ``think`` defaults to ``False`` because Yasama/Yurutme use small,
    deterministic output budgets.  Set it to ``True`` (or a supported Ollama
    thinking level string) when constructing the backend if a model/role should
    use reasoning output.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        think: bool | str | None = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.think = think

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        return self._generate(messages, config=config)

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, Any],
        config: GenerationConfig | None = None,
    ) -> str:
        """Request Ollama-native JSON Schema constrained output."""
        if schema.get("type") != "object":
            raise ValueError("Structured Ollama output requires an object JSON schema.")
        return self._generate(messages, config=config, schema=schema)

    def _generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        config: GenerationConfig | None = None,
        schema: Mapping[str, Any] | None = None,
    ) -> str:
        cfg = config or GenerationConfig()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _normalized_messages(messages),
            "stream": False,
            "think": self.think,
            "options": {
                "num_predict": cfg.max_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "repeat_penalty": cfg.repeat_penalty,
                "repeat_last_n": cfg.repeat_last_n,
            },
        }
        if schema is not None:
            payload["format"] = dict(schema)
        response = _post_json(
            f"{self.base_url}/api/chat",
            payload,
            timeout=cfg.timeout_seconds,
        )
        message = response.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise RuntimeError("Ollama response is missing message.content.")

        content = str(message["content"])
        if not content.strip():
            thinking = message.get("thinking")
            thinking_detail = (
                f" thinking_chars={len(thinking)}" if isinstance(thinking, str) and thinking else ""
            )
            done_reason = response.get("done_reason")
            reason_detail = f" done_reason={done_reason!r}" if done_reason is not None else ""
            raise RuntimeError(
                "Ollama returned empty message.content."
                + thinking_detail
                + reason_detail
                + " Try think=false or increase the generation budget."
            )
        return content
