"""Yasama runtime: chooses the next canonical action only."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from runtime.backends import ChatBackend, GenerationConfig
from runtime.local_tools import BUILTIN_TOOL_REGISTRY
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
        "CALCULATE",
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

_YASAMA_PROMPT_TEMPLATE = """Sen Mercan Yasama modelisin.
Görevin kullanıcı isteği, konuşma geçmişi ve mevcut state'e bakarak yalnızca bir sonraki canonical action'ı seçmektir.
Argüman üretme, açıklama yapma ve kullanıcıya final cevap yazma.
Çıktın yalnızca aşağıdaki action isimlerinden tam olarak biri olmalıdır:
{action_names}.

En üstteki USER alanı mevcut istektir ve her zaman birincildir. CONVERSATION_HISTORY yalnızca bağlamdır; önceki isteğin action'ını yeni USER isteğine kopyalama.

{action_descriptions}
Niyet eşleme örnekleri:
- Kullanıcı sadece sohbet/bilgi cevabı istiyorsa ve araç gerekmiyorsa: RESPOND
- Kullanıcı e-posta/mail göndermek istiyor ama alıcı, konu veya içerik eksikse: ASK_CLARIFICATION
- Mail gönderimi için alıcı adresi, konu ve mesaj içeriği (ya da yeterli içerik talimatı) açıkça belli olduğunda: SEND_MAIL
- Kullanıcı klasördeki dosyaları listelemek istiyorsa: FILE_LIST
- Kullanıcı bir dosyanın içeriğini okumak istiyorsa: FILE_READ
- Kullanıcı dosyalar içinde metin aramak istiyorsa: FILE_SEARCH
- Kullanıcı sayılarla bir hesap (toplama, çıkarma, çarpma, bölme, üs, karekök) istiyorsa: CALCULATE
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

_CONTROL_ACTION_DESCRIPTIONS = {
    "RESPOND": "Araç gerekmeden doğrudan cevap verilecek sohbet veya bilgi soruları.",
    "ASK_CLARIFICATION": (
        "İstenen araç için zorunlu bilgi (örneğin mail alıcısı) eksikse "
        "kullanıcıya soru sorulur."
    ),
    "FINISH": "Mevcut turda gerekli araç çalıştı ve sonucu STATE.recent_observations içinde.",
}
_ACTION_ORDER = (
    "RESPOND", "WEB_SEARCH", "WEB_OPEN", "FILE_READ", "FILE_SEARCH", "FILE_LIST",
    "CALCULATE", "FILE_CREATE", "FILE_EDIT", "FILE_DELETE", "FILE_MOVE", "CODE_EXECUTE",
    "SHELL_EXECUTE", "DOWNLOAD", "SEND_MAIL", "ASK_CLARIFICATION", "FINISH",
)
assert frozenset(_ACTION_ORDER) == CANONICAL_ACTIONS


def _action_descriptions() -> str:
    """Describe actions from the same registry Yurutme receives schemas from."""
    lines = ["Action'lar ve ne zaman seçilecekleri:"]
    for action in _ACTION_ORDER:
        description = _CONTROL_ACTION_DESCRIPTIONS.get(action)
        if description is None:
            description = BUILTIN_TOOL_REGISTRY[action]["description"]
        lines.append(f"- {action}: {description}")
    return "\n".join(lines) + "\n"


YASAMA_SYSTEM_PROMPT = _YASAMA_PROMPT_TEMPLATE.replace(
    "{action_names}", ", ".join(_ACTION_ORDER)
).replace("{action_descriptions}", _action_descriptions())

_EMPTY_STATE: dict[str, Any] = {
    "completed_actions": [],
    "last_observation": None,
    "recent_observations": [],
    "step_index": 0,
    "turn_index": 0,
}
_LIST_OBSERVATION: dict[str, Any] = {
    "action": "FILE_LIST",
    "data": {"entries": ["docs/api.md"]},
    "status": "success",
    "error": None,
}
_READ_OBSERVATION: dict[str, Any] = {
    "action": "FILE_READ",
    "data": {"path": "README.md"},
    "status": "success",
    "error": None,
}
_CALCULATE_OBSERVATION: dict[str, Any] = {
    "action": "CALCULATE",
    "data": {"result": 156.25},
    "status": "success",
    "error": None,
}

# Shown to the model as prior user/assistant turns rather than inline in the
# system prompt: small checkpoints echo inline examples instead of following
# them, but answer chat-turn examples with a bare action name.
YASAMA_FEW_SHOT_EXAMPLES: tuple[tuple[str, list[dict[str, str]], dict[str, Any], str], ...] = (
    ("Merhaba, nasılsın?", [], _EMPTY_STATE, "RESPOND"),
    ("Bu klasördeki Python dosyalarını listele.", [], _EMPTY_STATE, "FILE_LIST"),
    ("README.md dosyasını okuyup özetle.", [], _EMPTY_STATE, "FILE_READ"),
    ('Projede "GenerationConfig" geçen yerleri bul.', [], _EMPTY_STATE, "FILE_SEARCH"),
    ("İnternette Ankara hava durumunu ara.", [], _EMPTY_STATE, "WEB_SEARCH"),
    ("git status komutunu çalıştır.", [], _EMPTY_STATE, "SHELL_EXECUTE"),
    ("print(2+2) kodunu çalıştır.", [], _EMPTY_STATE, "CODE_EXECUTE"),
    ("Mail göndermek istiyorum.", [], _EMPTY_STATE, "ASK_CLARIFICATION"),
    (
        'ali@ornek.com adresine "Toplantı" konulu, "Yarın 10\'da görüşelim." '
        "içerikli bir mail gönder.",
        [], _EMPTY_STATE, "SEND_MAIL",
    ),
    # After the original examples: placed earlier, it moved unrelated
    # decisions of the untrained checkpoint (a greeting became WEB_SEARCH).
    ("1250 lirayı 8 kişiye eşit bölersek kişi başı ne düşer?", [], _EMPTY_STATE, "CALCULATE"),
    (
        "README.md dosyasını okuyup özetle.",
        [],
        {
            **_EMPTY_STATE,
            "completed_actions": ["FILE_READ"],
            "last_observation": _READ_OBSERVATION,
            "recent_observations": [_READ_OBSERVATION],
            "step_index": 1,
        },
        "FINISH",
    ),
    (
        "docs klasöründeki dosyaları listele.",
        [],
        {
            **_EMPTY_STATE,
            "completed_actions": ["FILE_LIST"],
            "last_observation": _LIST_OBSERVATION,
            "recent_observations": [_LIST_OBSERVATION],
            "step_index": 1,
        },
        "FINISH",
    ),
    (
        "1250 lirayı 8 kişiye eşit bölersek kişi başı ne düşer?",
        [],
        {
            **_EMPTY_STATE,
            "completed_actions": ["CALCULATE"],
            "last_observation": _CALCULATE_OBSERVATION,
            "recent_observations": [_CALCULATE_OBSERVATION],
            "step_index": 1,
        },
        "FINISH",
    ),
    (
        "Türkiye'nin başkenti neresi?",
        [
            {"role": "user", "content": "Bu klasördeki dosyaları listele."},
            {"role": "assistant", "content": "Klasörde main.py ve README.md var."},
        ],
        _EMPTY_STATE,
        "RESPOND",
    ),
)


def _few_shot_messages() -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for user_prompt, history, state, action in YASAMA_FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": serialize_yasama_input(
                user_prompt=user_prompt,
                conversation_history=history,
                state=state,
            ),
        })
        messages.append({"role": "assistant", "content": action})
    return messages

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
                *_few_shot_messages(),
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
