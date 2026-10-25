"""Yargi runtime: composes a grounded user-facing response from real observations."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from runtime.backends import ChatBackend, GenerationConfig
from runtime.output_parsing import trim_trailing_loop, unwrap_natural_language_response
from runtime.serialization import serialize_yargi_input


YARGI_SYSTEM_PROMPT = """Sen Yargı modelisin. Kullanıcının mevcut isteğini ve gerçek araç gözlemlerini kullanarak nihai cevabı üret.
Gözlemlerde bulunmayan bilgi uydurma. Executor hata döndürdüyse başarı iddiasında bulunma.

Son user mesajının ilk satırındaki mevcut istek her zaman birincildir. Önceki konuşma turları yalnızca bağlamdır; önceki bir isteğin cevabını yeni isteğe kopyalama.
<tool_observations> boş değilse dosya yolu, içerik, arama sonucu, komut sonucu veya işlem başarısı gibi araçla ilgili tüm olguları yalnızca bu gözlemlerden al. Gözlemde olmayan klasör, dosya konumu veya sonuç uydurma.
<tool_observations> boşsa önceki araç sonuçlarını anlatma; yalnız mevcut kullanıcı isteğine cevap ver.
RUNTIME_INFO verilmişse, sistemde hangi rollerin ve araçların bulunduğuna dair sorularda yalnız o bilgileri kullan. Kurulu olmayan veya yapılandırılmamış araçların çalıştığını iddia etme.
Kullanıcı yeni bir soru sorduğunda önceden verilen yanıtı tekrarlama. FINISH, RESPOND gibi action adlarını kullanıcıya açıklama olarak yazma.

Çıktın doğrudan kullanıcıya gidecek doğal dil cevabı olmalıdır. JSON object, JSON wrapper, markdown code fence veya alan adı (response, result, file_summary vb.) kullanma.
"""


# The prompt a fine-tuned Yargi sees, in training data and at runtime
# (prompt_style "trained").
TRAINED_SYSTEM_PROMPT = """Sen Mercan'ın Yargı modelisin. Kullanıcının isteğine, <tool_observations> içindeki gerçek araç sonuçlarına dayanarak kullanıcıya gidecek Türkçe cevabı yazarsın.
- Sayı, dosya, bağlantı ve işlem sonucu gibi araçla ilgili her bilgiyi yalnızca gözlemlerden al; gözlemde olmayanı uydurma.
- Bir araç hata verdiyse veya sonuç boşsa bunu açıkça söyle; başarılı olmuş gibi anlatma.
- <tool_observations> boşsa isteğe doğrudan cevap ver.
- Önceki turlar yalnızca bağlamdır; önceki cevabı tekrarlama.
- Ham JSON, alan adları (action, status, data) veya iç adım adları yazma; sonucu doğal dille, açık ve öz anlat."""


def system_message(*, trained: bool = True, runtime_info: Mapping[str, Any] | None = None) -> dict[str, str]:
    content = TRAINED_SYSTEM_PROMPT if trained else YARGI_SYSTEM_PROMPT
    if runtime_info:
        content += "\nRUNTIME_INFO:\n" + json.dumps(dict(runtime_info), ensure_ascii=False)
    return {"role": "system", "content": content}


def _observation(
    action: str, data: dict[str, Any], error: str | None = None,
) -> dict[str, Any]:
    return {
        "action": action,
        "data": data,
        "status": "error" if error else "success",
        "error": error,
    }


# Shown as prior user/assistant turns.  Inline system-prompt examples made
# small checkpoints echo the observations or the rules themselves.  The set
# mixes successes, an error and a no-tool turn so the model does not copy one
# outcome (e.g. reporting "not found" for a successful read).
YARGI_FEW_SHOT_EXAMPLES: tuple[tuple[str, list[dict[str, Any]], str], ...] = (
    (
        "docs klasöründeki dosyaları listele.",
        [_observation("FILE_LIST", {
            "path": "docs", "pattern": "*", "recursive": False,
            "entries": ["docs/kurulum.md", "docs/api.md", "docs/sss.md", "docs/logo.png"],
        })],
        "docs klasöründe dört dosya var: kurulum.md, api.md, sss.md ve logo.png.",
    ),
    (
        "VERSION dosyasını oku, sürüm ne?",
        [_observation("FILE_READ", {"path": "VERSION", "content": "2.4.1\n"})],
        "VERSION dosyasına göre sürüm 2.4.1.",
    ),
    (
        "189 km'yi 3 saatte giden aracın saatteki hızı kaç?",
        [_observation("CALCULATE", {"result": 63})],
        "Aracın saatteki ortalama hızı 63 km.",
    ),
    (
        "Projede 'TODO' geçen yerleri bul.",
        [_observation("FILE_SEARCH", {}, "Directory not found: /src")],
        "Arama yapılamadı; aranacak klasör bulunamadı.",
    ),
    (
        "Merhaba, nasılsın?",
        [],
        "Merhaba, iyiyim, teşekkürler. Sana nasıl yardımcı olabilirim?",
    ),
)


def _few_shot_messages() -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for user_prompt, observations, response in YARGI_FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": serialize_yargi_input(
                user_prompt=user_prompt,
                conversation_history=[],
                observations=observations,
            ),
        })
        messages.append({"role": "assistant", "content": response})
    return messages


YARGI_HISTORY_MESSAGES = 6


def _history_messages(
    conversation_history: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Replay recent turns as real chat turns.

    Embedded as a CONVERSATION_HISTORY JSON block, small checkpoints echoed
    the block as the answer, and the echo then grew the history every turn.
    """
    turns: list[dict[str, str]] = []
    for message in conversation_history:
        role, content = message.get("role"), message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        if role == "user":
            # Same shape as the current request: past turns carry no tool
            # observations of their own.
            content = serialize_yargi_input(
                user_prompt=content, conversation_history=[], observations=[],
            )
        turns.append({"role": role, "content": content})
    return turns[-YARGI_HISTORY_MESSAGES:]


InferenceFn = Callable[[dict[str, Any]], str]


def request_messages(
    *,
    user_prompt: str,
    conversation_history: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Recent history turns plus the request turn, shared by inference and
    training data (datatools)."""
    return [
        *_history_messages(conversation_history),
        {
            "role": "user",
            "content": serialize_yargi_input(
                user_prompt=user_prompt,
                conversation_history=[],
                observations=observations,
            ),
        },
    ]


# Trained-checkpoint generation (default_generation_config): greedy, as
# training/eval_role.py measures it. 1024 tokens cover the test answers' p99
# (~2.7k characters). No repeat penalty: 1.05/1.1 raised answers with
# ungrounded numbers from 10.6% to 13.2%/14.3%; trim_trailing_loop cuts loops.
TRAINED_MAX_TOKENS = 1024
TRAINED_REPEAT_PENALTY = 1.0


def default_generation_config(*, trained: bool = False) -> GenerationConfig:
    if trained:
        return GenerationConfig(max_tokens=TRAINED_MAX_TOKENS, temperature=0.0, top_p=1.0, top_k=1,
                                repeat_penalty=TRAINED_REPEAT_PENALTY, repeat_last_n=64)
    return GenerationConfig(max_tokens=768, temperature=0.2, top_p=0.95, top_k=40, repeat_penalty=1.05)


class YargiRuntime:
    """Owns Yargi inference and returns the final natural-language response."""

    def __init__(
        self,
        inference_fn: InferenceFn | None = None,
        *,
        backend: ChatBackend | None = None,
        generation_config: GenerationConfig | None = None,
        runtime_info: Mapping[str, Any] | None = None,
        trained: bool = False,
    ) -> None:
        if inference_fn is not None and backend is not None:
            raise ValueError("Configure either inference_fn or backend, not both.")
        self._inference_fn = inference_fn
        self._backend = backend
        # A fine-tuned checkpoint gets its training prompt and no few-shot turns.
        self._trained = trained
        self._runtime_info = dict(runtime_info or {})
        self._generation_config = generation_config or default_generation_config(trained=trained)

    def set_runtime_info(self, info: Mapping[str, Any]) -> None:
        """Supply actual configured role and tool information, not model guesses."""
        self._runtime_info = dict(info)

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

        # Greedy decoding can loop until max_tokens; a repeat penalty would
        # also push grounded numbers away (PROJECT_MASTER §12).
        response = trim_trailing_loop(unwrap_natural_language_response(self._infer(payload)))
        if not response:
            raise ValueError("Yargi produced an empty response.")

        return response

    def _infer(self, payload: dict[str, Any]) -> str:
        if self._inference_fn is not None:
            return self._inference_fn(payload)

        if self._backend is not None:
            turns = request_messages(
                user_prompt=payload["user_prompt"],
                conversation_history=payload["conversation_history"],
                observations=payload["observations"],
            )
            messages = [
                # The trained checkpoint never saw RUNTIME_INFO; with it, it
                # recited local model paths as "tools" (PROJECT_MASTER §12).
                system_message(trained=self._trained,
                               runtime_info=None if self._trained else self._runtime_info),
                # With real history the model cannot tell example turns from
                # the conversation and repeats example observations as facts.
                *([] if self._trained or turns[:-1] else _few_shot_messages()),
                *turns,
            ]
            return self._backend.generate(messages, config=self._generation_config)

        raise RuntimeError(
            "Yargi model runtime is not configured. "
            "Pass an inference_fn or a ChatBackend."
        )
