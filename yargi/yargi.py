"""Yargi runtime: composes a grounded user-facing response from real observations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from runtime.backends import ChatBackend, GenerationConfig
from runtime.output_parsing import unwrap_natural_language_response
from runtime.serialization import serialize_yargi_input


YARGI_SYSTEM_PROMPT = """Sen Yargı modelisin. Kullanıcının mevcut isteğini ve gerçek araç gözlemlerini kullanarak nihai cevabı üret.
Gözlemlerde bulunmayan bilgi uydurma. Executor hata döndürdüyse başarı iddiasında bulunma.

User mesajının ilk satırındaki mevcut istek her zaman birincildir. CONVERSATION_HISTORY yalnızca bağlamdır; önceki bir isteğin cevabını yeni isteğe kopyalama.
<tool_observations> boş değilse dosya yolu, içerik, arama sonucu, komut sonucu veya işlem başarısı gibi araçla ilgili tüm olguları yalnızca bu gözlemlerden al. Gözlemde olmayan klasör, dosya konumu veya sonuç uydurma.
<tool_observations> boşsa önceki araç sonuçlarını anlatma; yalnız mevcut kullanıcı isteğine cevap ver.

Çıktın doğrudan kullanıcıya gidecek doğal dil cevabı olmalıdır. JSON object, JSON wrapper, markdown code fence veya alan adı (response, result, file_summary vb.) kullanma.
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

        response = unwrap_natural_language_response(self._infer(payload))
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
                    "content": serialize_yargi_input(
                        user_prompt=payload["user_prompt"],
                        conversation_history=payload["conversation_history"],
                        observations=payload["observations"],
                    ),
                },
            ]
            return self._backend.generate(messages, config=self._generation_config)

        raise RuntimeError(
            "Yargi model runtime is not configured. "
            "Pass an inference_fn or a ChatBackend."
        )
