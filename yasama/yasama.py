"""Yasama runtime: chooses the next canonical action only."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from runtime.backends import ChatBackend, GenerationConfig
from runtime.output_parsing import parse_canonical_action
from runtime.serialization import serialize_yasama_input


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

YASAMA_SYSTEM_PROMPT = """Sen Mercan Yasama modelisin.
Görevin kullanıcı isteği, konuşma geçmişi ve mevcut state'e bakarak yalnızca bir sonraki canonical action'ı seçmektir.
Argüman üretme, açıklama yapma ve kullanıcıya final cevap yazma.
Çıktın yalnızca aşağıdaki action isimlerinden tam olarak biri olmalıdır:
RESPOND, WEB_SEARCH, WEB_OPEN, FILE_READ, FILE_SEARCH, FILE_LIST, FILE_CREATE, FILE_EDIT, FILE_DELETE, FILE_MOVE, CODE_EXECUTE, SHELL_EXECUTE, DOWNLOAD, SEND_MAIL, ASK_CLARIFICATION, FINISH.

En üstteki USER alanı mevcut istektir ve her zaman birincildir. CONVERSATION_HISTORY yalnızca bağlamdır; önceki isteğin action'ını yeni USER isteğine kopyalama.

Niyet eşleme örnekleri:
- Kullanıcı sadece sohbet/bilgi cevabı istiyorsa ve araç gerekmiyorsa: RESPOND
- Kullanıcı e-posta/mail göndermek istiyor ama alıcı, konu veya içerik eksikse: ASK_CLARIFICATION
- Mail gönderimi için alıcı adresi, konu ve mesaj içeriği (ya da yeterli içerik talimatı) açıkça belli olduğunda: SEND_MAIL
- Kullanıcı klasördeki dosyaları listelemek istiyorsa: FILE_LIST
- Kullanıcı bir dosyanın içeriğini okumak istiyorsa: FILE_READ
- Kullanıcı dosyalar içinde metin aramak istiyorsa: FILE_SEARCH
- Kullanıcı internette arama yapmak istiyorsa: WEB_SEARCH
- Kullanıcı belirli bir web sayfasını açmak istiyorsa: WEB_OPEN
- Kullanıcı terminal/shell komutunu doğrudan çalıştırmak istiyorsa: SHELL_EXECUTE
- Kullanıcı Python/kod parçasını çalıştırmak istiyorsa: CODE_EXECUTE
- Gerekli araç işleri başarıyla bittiyse: FINISH

Önemli karar kuralları:
- FINISH yalnızca mevcut turdaki STATE.recent_observations alanında bir araç gözlemi varsa seçilebilir. Önceki konuşmanın bitmesi, mevcut turun FINISH olması anlamına gelmez.
- Önceki konuşmadaki eylem veya cevap, kullanıcının şu anki isteğine dönüşmemelidir.
- Kullanıcı yeni bir bilgi sorusu soruyorsa veya sohbet ediyorsa RESPOND seç.
- Araç çağrısı için zorunlu bilgiler eksikse ASK_CLARIFICATION seç; eksik verileri uydurup aracı çağırma.
- Yan etkili eylemler (örneğin mail gönderme) yalnızca kullanıcının mevcut talebiyle gerekçelendirilebilir.
"ACTION:" etiketi, JSON, markdown veya açıklama yazma. Yalnız action adını yaz.
"""

InferenceFn = Callable[[dict[str, Any]], str]


class YasamaRuntime:
    """Owns Yasama inference and validates its single-action output."""

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
            max_tokens=16,
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
    ) -> str:
        payload = {
            "user_prompt": user_prompt,
            "conversation_history": list(conversation_history),
            "state": dict(state),
        }

        return parse_canonical_action(self._infer(payload), CANONICAL_ACTIONS)

    def _infer(self, payload: dict[str, Any]) -> str:
        if self._inference_fn is not None:
            return self._inference_fn(payload)

        if self._backend is not None:
            messages = [
                {"role": "system", "content": YASAMA_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": serialize_yasama_input(
                        user_prompt=payload["user_prompt"],
                        conversation_history=payload["conversation_history"],
                        state=payload["state"],
                    ),
                },
            ]
            return self._backend.generate(messages, config=self._generation_config)

        raise RuntimeError(
            "Yasama model runtime is not configured. "
            "Pass an inference_fn or a ChatBackend."
        )
