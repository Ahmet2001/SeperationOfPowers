"""Teacher-free records for the gaps in PROJECT_MASTER §7.

* RESPOND: Turkish chat requests that need no tool become Yasama RESPOND
  records, and their answers Yargi records with no observations (the
  runtime answers RESPOND with Yargi and an empty observation list).
* History mixing: existing records get an unrelated earlier exchange; the
  label stays. This trains exactly the two observed failures: Yasama
  copying the previous turn's action and Yargi echoing the history.

Rows are produced in the source-row shape the builders already read.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

EMPTY_STATE = {"completed_actions": [], "last_observation": None, "recent_observations": []}

# A chat request that matches any of these probably needs a tool (or is
# ambiguous) and is not used as a RESPOND example.
_TOOL_NEED = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\d\s*[-+*/x×÷^%]\s*\d",                                  # arithmetic
        r"\b(hesapla|kaç eder|toplam[ıi]|karekök|yüzde\s*\d|%\s*\d)",
        r"\b(bugün|bu gün|şu an|şimdi|güncel|son dakika|haber|hava durumu|kur[uı]?\b|dolar|euro|borsa|fiyat[ıi]? ne)",
        r"\b(20(2[4-9]|3\d))\b",                                    # recent years
        r"\b(internet|web|google|ara\b|arama yap|araştır|site|link|bağlantı)",
        r"https?://|www\.|\S+@\S+\.\w+",
        r"\.(py|js|ts|json|csv|txt|md|pdf|docx?|xlsx?|html|yaml|yml|sh)\b",
        r"\b(dosya|klasör|dizin|mail|e-posta|eposta|indir|yükle|çalıştır|terminal|komut|kodu çalıştır)",
    )
]


def needs_tool(prompt: str) -> bool:
    return any(p.search(prompt) for p in _TOOL_NEED)


def stable_share(*parts: Any) -> float:
    """Deterministic [0, 1) value for sampling without a random seed."""
    return int(hashlib.md5(":".join(map(str, parts)).encode()).hexdigest(), 16) / 16**32


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def respond_rows(
    source_id: str, prompt: str, answer: str | None, history: Sequence[Mapping[str, str]] = (),
    *, source_dataset: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Yasama (and, with an answer, Yargi) rows for a request with no tool."""
    rows = {"yasama": {
        "source_id": source_id, "source_dataset": source_dataset, "step_index": 0, "user_prompt": prompt,
        "conversation_history": list(history), "state": dict(EMPTY_STATE), "target_action": "RESPOND",
    }}
    if answer:
        rows["yargi"] = {
            "source_id": source_id, "source_dataset": source_dataset, "user_prompt": prompt,
            "conversation_history": list(history),
            "observations": [], "response": _strip_thinking(answer), "respond": True,
        }
    return rows


def dpo_chat(row: Mapping[str, Any]) -> tuple[str, str] | None:
    prompt, chosen = row.get("prompt"), row.get("chosen")
    if isinstance(prompt, str) and isinstance(chosen, str) and prompt.strip() and chosen.strip():
        return prompt.strip(), chosen.strip()
    return None


HELPSTEER_DOMAINS = frozenset({"general", "stem"})  # code translations were garbled


def helpsteer_chat(row: Mapping[str, Any]) -> tuple[list[dict[str, str]], str, str] | None:
    """(history, last user request, preferred response). HelpSteer3's
    overall_preference is negative when response1 is better."""
    if row.get("domain") not in HELPSTEER_DOMAINS:
        return None
    context = [m for m in row.get("context") or [] if isinstance(m, Mapping)]
    if not context or context[-1].get("role") != "user":
        return None
    preference = row.get("overall_preference")
    if not isinstance(preference, (int, float)) or preference == 0:
        return None
    answer = row.get("response1") if preference < 0 else row.get("response2")
    if not isinstance(answer, str) or not answer.strip():
        return None
    history = [{"role": m["role"], "content": m["content"]} for m in context[:-1]
               if m.get("role") in {"user", "assistant"} and isinstance(m.get("content"), str)]
    return history, str(context[-1]["content"]), answer


# ---------------------------------------------------------- history mixing

# Classes with hundreds of thousands of records contribute a small share;
# every rarer class is mixed in full.
BIG_ACTIONS = frozenset({"WEB_SEARCH", "CALCULATE", "CODE_EXECUTE", "FINISH", "FILE_CREATE"})
BIG_RATE, OPEN_TOOL_RATE, RARE_RATE = 0.004, 0.1, 1.0
MAX_HISTORY_ANSWER = 1200


def history_rate(label: str) -> float:
    if label == "OPEN_TOOL":
        return OPEN_TOOL_RATE
    return BIG_RATE if label in BIG_ACTIONS else RARE_RATE


def exchange(user: str, assistant: str) -> list[dict[str, str]] | None:
    """One earlier turn for a history pool; long answers are skipped rather
    than cut, so the history reads like a real conversation."""
    if not user.strip() or not assistant.strip() or len(assistant) > MAX_HISTORY_ANSWER:
        return None
    return [{"role": "user", "content": user.strip()}, {"role": "assistant", "content": assistant.strip()}]


def pick(pool: Sequence[Any], *key: Any) -> Any:
    return pool[int(stable_share("pick", *key) * len(pool))]


def with_history(row: Mapping[str, Any], earlier: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    mixed = dict(row)
    mixed["conversation_history"] = [*earlier, *(row.get("conversation_history") or [])]
    mixed["source_id"] = f"{row.get('source_id')}#hist"
    return mixed


DPO_DATASET = "erenfazlioglu/turkish-instruct-reasoning-dpo-3.4m"
HUMAN_LIKE_DATASET = "Human-Like-DPO-Dataset-TR"
HELPSTEER_DATASET = "yusufbaykaloglu/helpsteer3-tr"

# The Mercan DPO model was trained on dpo.parquet, HelpSteer3 train and
# Human-Like (continuation_dpo_v1). Yasama only takes their requests with a
# new label, so all of them are used. Yargi would be fine-tuned again on
# answers the model was already pushed towards, so it gets a few thousand
# short ones (train only) to learn the empty-observation format, and its
# RESPOND evaluation comes from HelpSteer3 validation, which DPO held out.
YARGI_DPO_RATE = 0.03
MAX_YARGI_RESPOND_ANSWER = 1200
HOLDOUT_SPLITS = ((0.5, "train"), (0.75, "validation"), (1.0, "test"))


def holdout_split(source_id: str) -> str:
    share = stable_share("holdout", source_id)
    return next(split for bound, split in HOLDOUT_SPLITS if share < bound)


def chat_respond_rows(
    role: str, dpo_rows: Iterator[Mapping[str, Any]], human_like_rows: Iterator[Mapping[str, Any]],
    helpsteer_rows: Iterator[Mapping[str, Any]], helpsteer_holdout_rows: Iterator[Mapping[str, Any]], *,
    dpo_rate: float, yargi_dpo_rate: float = YARGI_DPO_RATE,
) -> Iterator[dict[str, Any]]:
    """RESPOND rows for one role from the chat sources (see above).

    Human-Like-DPO answers claim personal experiences, so only its requests
    are used. Rows with a fixed split carry it in ``_split``.
    """
    if role == "yasama":
        for index, row in enumerate(dpo_rows):
            chat = dpo_chat(row)
            if chat and not needs_tool(chat[0]) and stable_share("dpo", index) < dpo_rate:
                yield respond_rows(f"dpo-{index}", chat[0], None, source_dataset=DPO_DATASET)["yasama"]
        for index, row in enumerate(human_like_rows):
            chat = dpo_chat(row)
            if chat and not needs_tool(chat[0]):
                yield respond_rows(f"humanlike-{index}", chat[0], None, source_dataset=HUMAN_LIKE_DATASET)["yasama"]
        for index, row in enumerate(helpsteer_rows):
            chat = helpsteer_chat(row)
            if chat and not needs_tool(chat[1]):
                yield respond_rows(f"helpsteer-{index}", chat[1], None, chat[0],
                                   source_dataset=HELPSTEER_DATASET)["yasama"]
        return
    if role != "yargi":
        return
    for index, row in enumerate(dpo_rows):
        chat = dpo_chat(row)
        if (chat and not needs_tool(chat[0]) and stable_share("dpo-yargi", index) < yargi_dpo_rate
                and len(_strip_thinking(chat[1])) <= MAX_YARGI_RESPOND_ANSWER):
            record = respond_rows(f"dpo-{index}", chat[0], chat[1], source_dataset=DPO_DATASET)["yargi"]
            yield {**record, "_split": "train"}
    for index, row in enumerate(helpsteer_holdout_rows):
        chat = helpsteer_chat(row)
        if chat and not needs_tool(chat[1]):
            source_id = f"helpsteer-validation-{index}"
            record = respond_rows(source_id, chat[1], chat[2], chat[0], source_dataset=HELPSTEER_DATASET)["yargi"]
            yield {**record, "_split": holdout_split(source_id)}
