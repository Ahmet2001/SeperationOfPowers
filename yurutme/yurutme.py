"""Yurutme runtime: fills arguments for the action selected by Yasama."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from runtime.backends import ChatBackend, GenerationConfig
from runtime.local_tools import BUILTIN_TOOL_REGISTRY
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
"""

# Shown as prior user/assistant turns built from the built-in tool schemas.
# Each request gets the examples for its own action plus one contrasting
# action, which keeps the prompt near 1.6k tokens; inline system-prompt
# examples were copied verbatim by small checkpoints instead of followed.
# FILE_LIST and WEB_SEARCH carry two examples that differ in every value:
# with a single example the model copied its values whatever they were, and
# with none it copied the contrasting example (a "python" search query).
# FILE_EDIT has none on purpose: its example was copied into a wrong edit,
# while without one the model fails with invalid JSON and stops the pipeline.
YURUTME_FEW_SHOT_EXAMPLES: dict[str, tuple[tuple[str, dict[str, Any]], ...]] = {
    "FILE_LIST": (
        (
            "Bu klasördeki Python dosyalarını listele.",
            {"path": ".", "pattern": "*.py", "recursive": False},
        ),
        (
            "docs klasöründeki dosyaları listele.",
            {"path": "docs", "pattern": "*", "recursive": False},
        ),
    ),
    "FILE_READ": (
        (
            "README.md dosyasını okuyup özetle.",
            {"path": "README.md"},
        ),
        (
            "kurulum.md dosyasında Docker kurulumu nasıl anlatılıyor?",
            {"path": "kurulum.md", "query": "Docker kurulumu"},
        ),
    ),
    # Three different shapes so no single operator is copied; a decimal comma
    # and numbers written in words become plain digits, as in the data.
    "CALCULATE": (
        (
            "Bir koşucu 42,195 km'yi 3,5 saatte koştu. Saatteki ortalama hızı nedir?",
            {"expression": "42.195/3.5"},
        ),
        (
            "3 defter 12,50 TL'den, 2 kalem 4 TL'den alırsam toplam kaç TL öderim?",
            {"expression": "3*12.5+2*4"},
        ),
        (
            "Yüz kırk dördün karekökü kaç?",
            {"expression": "sqrt(144)"},
        ),
    ),
    "FILE_SEARCH": ((
        'Projede "GenerationConfig" geçen yerleri bul.',
        {"query": "GenerationConfig", "path": ".", "pattern": "*",
         "recursive": True, "max_results": 50},
    ),),
    "FILE_MOVE": ((
        "rapor.pdf dosyasını belgeler klasörüne taşı.",
        {"source": "rapor.pdf", "destination": "belgeler/rapor.pdf"},
    ),),
    "WEB_SEARCH": (
        (
            "İnternette 2026 Dünya Kupası maç takvimini ara.",
            {"query": "2026 Dünya Kupası maç takvimi", "max_results": 3},
        ),
        (
            "Webde Rust ile Python performans karşılaştırması bul.",
            {"query": "Rust Python performans karşılaştırması", "max_results": 3},
        ),
    ),
    "SHELL_EXECUTE": ((
        "git status komutunu çalıştır.",
        {"command": "git status", "cwd": "."},
    ),),
    "CODE_EXECUTE": ((
        "print(2+2) kodunu çalıştır.",
        {"language": "python", "code": "print(2+2)"},
    ),),
    "SEND_MAIL": ((
        'ali@ornek.com adresine "Toplantı" konulu, "Yarın 10\'da görüşelim." '
        "içerikli bir mail gönder.",
        {"to": "ali@ornek.com", "subject": "Toplantı",
         "body": "Yarın 10'da görüşelim."},
    ),),
}

_EMPTY_STATE: dict[str, Any] = {
    "completed_actions": [],
    "last_observation": None,
    "recent_observations": [],
    "step_index": 0,
    "turn_index": 0,
}


def _few_shot_messages(action: str) -> list[dict[str, str]]:
    contrast = "SHELL_EXECUTE" if action == "CODE_EXECUTE" else "CODE_EXECUTE"
    selected = [contrast] + ([action] if action in YURUTME_FEW_SHOT_EXAMPLES else [])
    messages: list[dict[str, str]] = []
    examples = [
        (example_action, user_prompt, arguments)
        for example_action in selected
        for user_prompt, arguments in YURUTME_FEW_SHOT_EXAMPLES[example_action]
    ]
    for example_action, user_prompt, arguments in examples:
        tool = BUILTIN_TOOL_REGISTRY[example_action]
        messages.append({
            "role": "user",
            "content": serialize_yurutme_input(
                user_prompt=user_prompt,
                conversation_history=[],
                state=_EMPTY_STATE,
                action=example_action,
                tool_schema={
                    key: value for key, value in tool.items()
                    if not key.startswith("_")
                },
                placeholder=tool.get("placeholder"),
            ),
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(arguments, ensure_ascii=False),
        })
    return messages


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
                *_few_shot_messages(payload["action"]),
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
