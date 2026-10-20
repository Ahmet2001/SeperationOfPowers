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
) -> dict[str, dict[str, Any]]:
    """Yasama (and, with an answer, Yargi) rows for a request with no tool."""
    rows = {"yasama": {
        "source_id": source_id, "step_index": 0, "user_prompt": prompt,
        "conversation_history": list(history), "state": dict(EMPTY_STATE), "target_action": "RESPOND",
    }}
    if answer:
        rows["yargi"] = {
            "source_id": source_id, "user_prompt": prompt, "conversation_history": list(history),
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


def chat_respond_rows(
    role: str, dpo_rows: Iterator[Mapping[str, Any]], human_like_rows: Iterator[Mapping[str, Any]],
    helpsteer_rows: Iterator[Mapping[str, Any]], *, dpo_rate: float,
) -> Iterator[dict[str, Any]]:
    """RESPOND rows for one role from the three chat sources.

    Human-Like-DPO answers claim personal experiences, so only its requests
    are used (Yasama routing); dpo and HelpSteer3 also give Yargi answers.
    """
    for index, row in enumerate(dpo_rows):
        chat = dpo_chat(row)
        if chat and not needs_tool(chat[0]) and stable_share("dpo", index) < dpo_rate:
            rows = respond_rows(f"dpo-{index}", chat[0], chat[1])
            if role in rows:
                yield rows[role]
    if role == "yasama":
        for index, row in enumerate(human_like_rows):
            chat = dpo_chat(row)
            if chat and not needs_tool(chat[0]):
                yield respond_rows(f"humanlike-{index}", chat[0], None)["yasama"]
    for index, row in enumerate(helpsteer_rows):
        chat = helpsteer_chat(row)
        if chat and not needs_tool(chat[1]):
            rows = respond_rows(f"helpsteer-{index}", chat[1], chat[2], chat[0])
            if role in rows:
                yield rows[role]
