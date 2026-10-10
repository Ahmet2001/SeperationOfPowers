"""Yurutme runtime: fills arguments for the action selected by Yasama."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from runtime.backends import ChatBackend, GenerationConfig
from runtime.output_parsing import parse_json_object
from runtime.tool_validation import validate_tool_arguments
from runtime.serialization import serialize_yurutme_input


YURUTME_SYSTEM_PROMPT = """Sen Mercan Yurutme modelisin.
Yasama tarafından seçilen action için kullanıcı isteği, konuşma geçmişi, state, tool schema ve placeholder'a bakarak doğru argüman JSON'unu üret.
Action seçme, kullanıcıya cevap verme ve açıklama ekleme.
Çıktın yalnızca geçerli bir JSON object olmalıdır; markdown code fence kullanma.

En üstteki USER alanı mevcut istektir ve her zaman birincildir. CONVERSATION_HISTORY yalnızca bağlamdır; önceki isteğin argümanlarını yeni USER isteğine kopyalama.
Kullanıcının belirttiği filtreleri koru. Örneğin "Python dosyalarını listele" ve FILE_LIST için pattern "*.py" olmalıdır; "JSON dosyaları" için "*.json" kullan.
PLACEHOLDER yalnızca beklenen alanların iskeletidir. Kullanıcı isteği daha spesifikse placeholder default'unu körü körüne kopyalama.
TOOL_SCHEMA / STATE / ACTION / USER / CONVERSATION_HISTORY anahtarlarını sonuç JSON'una kopyalama.
Örnek: USER: Projede "OllamaBackend" kelimesinin geçtiği Python dosyalarını ara. ACTION: FILE_SEARCH
Çıktı: {"query":"OllamaBackend","path":".","pattern":"*.py","recursive":true,"max_results":50}
"""

InferenceFn = Callable[[dict[str, Any]], Mapping[str, Any] | str]


class YurutmeRuntime:
    """Owns Yurutme inference and returns a JSON-compatible arguments object."""

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
            max_tokens=512,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            repeat_penalty=1.0,
        )

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

        correction: str | None = None
        attempts = 2 if self._backend is not None else 1
        for attempt in range(attempts):
            raw_arguments = self._infer(payload, correction=correction)
            try:
                if isinstance(raw_arguments, str):
                    raw_arguments = parse_json_object(raw_arguments)
                return validate_tool_arguments(
                    action, raw_arguments, payload["tool_schema"]
                )
            except (ValueError, TypeError) as exc:
                if attempt + 1 == attempts:
                    raise ValueError(
                        f"Yurutme could not produce valid {action} arguments: {exc}"
                    ) from exc
                correction = str(exc)

        raise AssertionError("Unreachable Yurutme retry state.")

    def _infer(
        self,
        payload: dict[str, Any],
        *,
        correction: str | None = None,
    ) -> Mapping[str, Any] | str:
        if self._inference_fn is not None:
            return self._inference_fn(payload)

        if self._backend is not None:
            messages = [
                {"role": "system", "content": YURUTME_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": serialize_yurutme_input(
                        user_prompt=payload["user_prompt"],
                        conversation_history=payload["conversation_history"],
                        state=payload["state"],
                        action=payload["action"],
                        tool_schema=payload["tool_schema"],
                        placeholder=payload["placeholder"],
                    ),
                },
            ]
            if correction is not None:
                messages.append({
                    "role": "user",
                    "content": (
                        "Önceki JSON yanlış: " + correction
                        + "\nYalnızca mevcut ACTION'ın TOOL_SCHEMA.parameters "
                        "alanına uygun argüman JSON nesnesini üret. "
                        "Gerekli alanları doldur; prompt'u veya tool_schema'yı "
                        "yanıta kopyalama."
                    ),
                })

            structured = getattr(self._backend, "generate_structured", None)
            parameters = payload["tool_schema"].get("parameters")
            if (
                callable(structured)
                and isinstance(parameters, Mapping)
                and parameters.get("type") == "object"
            ):
                return structured(
                    messages, schema=parameters, config=self._generation_config
                )
            return self._backend.generate(messages, config=self._generation_config)

        raise RuntimeError(
            "Yurutme model runtime is not configured. "
            "Pass an inference_fn or a ChatBackend."
        )
