"""Build a training mix from shared_data (PROJECT_MASTER §9.1).

    python -m datatools.build_mix \\
        --shared /arf/scratch/egitimg16/nedolm_0p8b/tooluse_inspect/yasama-yurutme-yargi/shared_data/v5 \\
        --out    /arf/scratch/egitimg16/nedolm_0p8b/tooluse_inspect/yasama-yurutme-yargi/mix/v1

shared_data keeps every record that meets the contract; a few sources make
up most of it (ms-marco web search, calculator chains, Jupyter code). The
mix keeps every record of the small classes, filters the big strata for
quality and takes a deterministic sample of them (lowest md5 of the record
id), so a rebuild picks the same records. Each record gets its role's
training system prompt, the one the runtime sends with prompt_style
"trained", as the first message.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from yargi.yargi import system_message as yargi_system_message
from yasama.yasama import system_message as yasama_system_message
from yurutme.yurutme import system_message as yurutme_system_message

MIX_VERSION = 1
ROLES = ("yasama", "yurutme", "yargi")
SPLITS = ("train", "validation", "test")
SYSTEM_MESSAGES = {
    "yasama": yasama_system_message(),
    "yurutme": yurutme_system_message(),
    "yargi": yargi_system_message(),
}
# Validation and test strata are capped at this share of the train cap: the
# eval sets keep the train mix's shape instead of the sources' skew.
EVAL_CAP_SHARE = 0.05

# (role, stratum) -> train cap. Strata not listed are kept whole.
CAPS: dict[tuple[str, str], int] = {
    ("yasama", "WEB_SEARCH/ms-marco"): 40_000,
    ("yasama", "FINISH/after-WEB_SEARCH"): 25_000,
    ("yasama", "FINISH/after-CALCULATE"): 15_000,
    ("yasama", "FINISH/after-CODE_EXECUTE"): 8_000,
    ("yasama", "CALCULATE/arac_kullanimi"): 30_000,
    ("yasama", "CODE_EXECUTE/combined_v2"): 10_000,
    ("yasama", "RESPOND/chat"): 40_000,
    # The query is the request verbatim in 99.6% of these.
    ("yurutme", "WEB_SEARCH/ms-marco"): 15_000,
    ("yurutme", "CALCULATE/arac_kullanimi"): 30_000,
    ("yurutme", "CODE_EXECUTE/combined_v2"): 10_000,
    ("yargi", "WEB_SEARCH/ms-marco"): 30_000,
    ("yargi", "CALCULATE/arac_kullanimi"): 20_000,
    ("yargi", "CODE_EXECUTE/combined_v2"): 10_000,
}

# ------------------------------------------------------------ text access

_YASAMA_INPUT = re.compile(r"USER:\n(.*?)\n(?:CONVERSATION_HISTORY:\n.*?\n)?STATE:\n(\{.*?\n\})", re.S)


def request_text(record: dict[str, Any]) -> str:
    return record["messages"][-2]["content"]


def user_prompt(role: str, record: dict[str, Any]) -> str:
    text = request_text(record)
    if role == "yargi":
        return text.split("\n<tool_observations>", 1)[0]
    match = _YASAMA_INPUT.match(text)
    return match.group(1) if match else text


def state(record: dict[str, Any]) -> dict[str, Any]:
    match = _YASAMA_INPUT.match(request_text(record))
    return json.loads(match.group(2)) if match else {}


def target(record: dict[str, Any]) -> str:
    return record["messages"][-1]["content"]


# ---------------------------------------------------------------- quality

_QUESTION = re.compile(
    r"\?\s*$|\b(nedir|nelerdir|nasıl|neden|niçin|niye|ne zaman|nerede|nereye|nereden|kim|kimdir|kaç|hangi|"
    r"ne kadar|ne demek|mi|mı|mu|mü|midir|mıdır|mudur|müdür)\b",
    re.IGNORECASE,
)


def search_prompt_ok(prompt: str) -> bool:
    """ms-marco requests worth keeping: a whole question, not a keyword
    fragment ("ms bir nörodejeneratif hastalık", "görüntü tanımı edebi")."""
    prompt = prompt.strip()
    return (20 <= len(prompt) <= 200 and (prompt[0].isupper() or prompt[0].isdigit())
            and bool(_QUESTION.search(prompt)))


def _stems(text: str) -> set[str]:
    # Five-letter prefixes stand in for Turkish stems (suffixes vary).
    return {w[:5] for w in re.findall(r"[\wçğıöşü]+", text.lower()) if len(w) >= 4 and not w.isdigit()}


def grounded_answer_ok(answer: str, observations: str) -> bool:
    """A whole sentence whose content words mostly come from the results:
    drops one-word fragments and translations that drifted from the source."""
    answer = answer.strip()
    if not 30 <= len(answer) <= 800 or answer[-1] not in ".!?)\"'":
        return False
    words = _stems(answer)
    return bool(words) and len(words & _stems(observations)) / len(words) >= 0.5


# --------------------------------------------------------------- strata

def stratum(role: str, record: dict[str, Any]) -> str | None:
    """The record's sampling stratum, or None when it fails a quality filter."""
    action, source = record["action"], record["source"]
    if source == "arac_kullanimi" and action == "WEB_SEARCH":
        if not search_prompt_ok(user_prompt(role, record)):
            return None
        if role == "yargi" and not grounded_answer_ok(target(record), request_text(record)):
            return None
        return "WEB_SEARCH/ms-marco"
    if role == "yasama" and action == "FINISH" and source in {"arac_kullanimi", "combined_v2"}:
        done = state(record).get("completed_actions") or []
        last = done[-1] if done else None
        if last == "WEB_SEARCH" and not search_prompt_ok(user_prompt(role, record)):
            return None  # same episodes as the WEB_SEARCH stratum
        if last in {"WEB_SEARCH", "CALCULATE", "CODE_EXECUTE"}:
            return f"FINISH/after-{last}"
    if source == "chat_respond" and role == "yasama":
        return "RESPOND/chat"
    return f"{action}/{source}"


def cap_for(role: str, name: str, split: str) -> int | None:
    cap = CAPS.get((role, name))
    if cap is None or split == "train":
        return cap
    return max(1, int(cap * EVAL_CAP_SHARE))


def sample_key(record: dict[str, Any]) -> str:
    return hashlib.md5(f"mix{MIX_VERSION}:{record['id']}".encode()).hexdigest()


# ----------------------------------------------------------------- build

def read_records(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def select(role: str, split: str, records: Callable[[], Iterator[dict[str, Any]]]):
    """Pass 1 keeps the lowest sample keys per capped stratum; pass 2 writes
    in source order. Returns (chosen ids, stratum counts before/after, rejects)."""
    heaps: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen: Counter = Counter()
    rejected: Counter = Counter()
    keep_all: set[str] = set()
    for record in records():
        name = stratum(role, record)
        if name is None:
            rejected[record["action"] + "/" + record["source"]] += 1
            continue
        seen[name] += 1
        cap = cap_for(role, name, split)
        if cap is None:
            keep_all.add(record["id"])
            continue
        # Negated keys make heap[0] the largest kept key: the one to replace.
        item = (-int(sample_key(record), 16), record["id"])
        heap = heaps[name]
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    chosen = keep_all | {record_id for heap in heaps.values() for _, record_id in heap}
    kept = Counter({name: len(heap) for name, heap in heaps.items()})
    for name in seen:
        if name not in kept:
            kept[name] = seen[name]
    return chosen, seen, kept, rejected


def build(shared: Path, out: Path, log: Callable[[str], None]) -> dict[str, Any]:
    shared_manifest = shared / "manifest.json"
    counts: dict[str, Any] = {}
    for role in ROLES:
        (out / role).mkdir(parents=True, exist_ok=True)
        counts[role] = {}
        for split in SPLITS:
            path = shared / role / f"{split}.jsonl.gz"
            chosen, seen, kept, rejected = select(role, split, lambda: read_records(path))
            system = SYSTEM_MESSAGES[role]
            actions: Counter = Counter()
            with gzip.open(out / role / f"{split}.jsonl.gz", "wt", encoding="utf-8") as writer:
                for record in read_records(path):
                    if record["id"] in chosen:
                        record["messages"] = [system, *record["messages"]]
                        writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                        actions[record["action"]] += 1
            counts[role][split] = {
                "records": sum(actions.values()),
                "by_action": dict(actions.most_common()),
                "strata": {name: {"available": seen[name], "kept": kept[name]}
                           for name in sorted(seen, key=lambda n: -seen[n])},
                "quality_rejects": dict(rejected.most_common()),
            }
            log(f"{role} {split}: {sum(actions.values())} records")
    return {
        "mix_version": MIX_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shared_data": str(shared),
        "shared_data_manifest_sha256": hashlib.sha256(shared_manifest.read_bytes()).hexdigest()
        if shared_manifest.exists() else None,
        "caps": {f"{role}:{name}": cap for (role, name), cap in CAPS.items()},
        "eval_cap_share": EVAL_CAP_SHARE,
        "system_prompts": {role: {"sha256": hashlib.sha256(m["content"].encode()).hexdigest(),
                                  "chars": len(m["content"])} for role, m in SYSTEM_MESSAGES.items()},
        "counts": counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shared", required=True, type=Path, help="shared_data/<version> directory")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--code-revision", help="SoP commit, when running from an archive without .git")
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    manifest = build(args.shared, args.out, log)
    manifest["code_revision"] = args.code_revision
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
