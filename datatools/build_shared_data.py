"""Build shared_data/<version> from the TRUBA sources (PROJECT_MASTER §9).

    python -m datatools.build_shared_data \\
        --root /arf/scratch/egitimg16/nedolm_0p8b/tooluse_inspect \\
        --out  /arf/scratch/egitimg16/nedolm_0p8b/tooluse_inspect/yasama-yurutme-yargi/shared_data/v1

Every record is normalized by ``datatools.normalize`` and serialized with the
roles' own request builders, so training text is the text the runtime sends
(§4). Records that cannot meet the contract are counted by reason, never
coerced. Output per role and split: ``<role>/<split>.jsonl.gz`` with
``messages`` (request turns + the assistant target, no system prompt) and
provenance fields that never enter the messages.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datatools.normalize import (
    CONTROL_ACTIONS,
    WRITE_AS_CREATE,
    Reject,
    check_record_length,
    canonical_action,
    check_calculation,
    check_answer_form,
    check_numbers_against_results,
    clean_response,
    load_aliases,
    new_file_write_steps,
    normalize_arguments,
    normalize_history,
    normalize_observation,
    normalize_state,
    open_tool_schema,
    validate_open_arguments,
)
from runtime.local_tools import BUILTIN_TOOL_REGISTRY
from runtime.registry_export import CONTRACT_VERSION, REGISTRY_PATH
from runtime.tool_validation import public_tool_schema
from yargi.yargi import request_messages as yargi_request_messages
from yasama.yasama import request_message as yasama_request_message
from datatools import short_calc, synth
from yurutme.yurutme import request_message as yurutme_request_message

DATASET_SCHEMA_VERSION = 6  # 2: open tools; 3: teacher-free RESPOND and history mixing;
                            # 4: Yargi RESPOND only from short dpo answers and HelpSteer3 validation;
                            # 5: Yasama STATE shows a data preview of the latest steps;
                            # 6: short_calc episodes; Yargi drops label / truncated / bare-number answers
ROLES = ("yasama", "yurutme", "yargi")
SPLITS = ("train", "validation", "test")
TEST_SHARE, VALIDATION_SHARE = 0.02, 0.02
REJECT_SAMPLES = 5
# Share of dpo chat requests used for Yasama RESPOND (~180k rows before
# filtering); Yargi uses synth.YARGI_DPO_RATE.
DPO_RESPOND_RATE = 0.2
# Earlier-turn pools for history mixing: sampling rates over the sources.
CHAT_EXCHANGE_RATE, TOOL_EXCHANGE_RATE = 0.03, 0.006


def sources(root: Path, role: str) -> list[tuple[str, Path]]:
    yyy = root / "yasama-yurutme-yargi"
    return [
        ("arac_kullanimi", root / "arac_kullanimi_v1" / "clean_yyy_v2" / f"{role}.arac_kullanimi.jsonl.gz"),
        *[("combined_v2", yyy / "combined_v2" / role / f"{split}.jsonl") for split in SPLITS],
        ("gemini_distil", yyy / role / "distil_data.jsonl"),
        ("hunter_tr", yyy / role / "hunter_tr.jsonl"),
    ]


def chat_paths(root: Path) -> dict[str, list[Path]]:
    dpo = root.parent / "post_sft_datasets_v1" / "dpo"
    extra = dpo / "general_preference_additions_v1"
    return {
        "dpo": [dpo / "dpo.parquet"],
        "human_like": sorted((extra / "Human-Like-DPO-Dataset-TR" / "data").glob("train-*.parquet")),
        "helpsteer": sorted((extra / "helpsteer3-tr" / "preference").glob("train-*.parquet")),
        # Held out of DPO training: the only chat answers Yargi has not seen.
        "helpsteer_holdout": sorted((extra / "helpsteer3-tr" / "preference").glob("validation-*.parquet")),
    }


def read_parquet_rows(paths: list[Path], limit: int | None = None) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq  # only the chat sources are parquet

    count = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                if limit is not None and count >= limit:
                    return
                count += 1
                yield row


def read_rows(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for count, line in enumerate(handle):
            if limit is not None and count >= limit:
                return
            if line.strip():
                yield json.loads(line)


def split_for(source: str, row: dict[str, Any]) -> str:
    # combined_v2 keeps its published split; everything else is split by
    # episode, so every step and role of one conversation lands together.
    if source == "combined_v2" and row.get("split") in SPLITS:
        return row["split"]
    key = f"{source}:{row.get('source_id')}".encode()
    share = int(hashlib.md5(key).hexdigest(), 16) / 16**32
    if share < TEST_SHARE:
        return "test"
    return "validation" if share < TEST_SHARE + VALIDATION_SHARE else "train"


# ------------------------------------------------------------ pre-passes

def collect_web_queries(root: Path, limit: int | None) -> dict[tuple[str, int], str]:
    """(source_id, step) -> WEB_SEARCH query from Yurutme rows; observations
    in the data carry no query of their own (§3.1 shape needs it)."""
    queries: dict[tuple[str, int], str] = {}
    for _, path in sources(root, "yurutme"):
        for row in read_rows(path, limit):
            args = row.get("target_arguments")
            if row.get("action") in {"web_search", "WEB_SEARCH"} and isinstance(args, dict):
                query = args.get("query") or args.get("q")
                if isinstance(query, str):
                    queries[(str(row.get("source_id")), int(row.get("step_index") or 0))] = query
    return queries


def collect_calc_results(root: Path, limit: int | None) -> tuple[dict[str, list[Any]], list[Any]]:
    """source_id -> calculator results per step, from arac_kullanimi Yargi rows;
    plus a sample of real tool turns (request + answer) for history mixing."""
    results: dict[str, list[Any]] = {}
    exchanges: list[Any] = []
    path = sources(root, "yargi")[0][1]
    for row in read_rows(path, limit):
        observations = row.get("observations") or []
        if observations and all(isinstance(o, dict) and o.get("action") in {"calc", "calculator"} for o in observations):
            results[str(row.get("source_id"))] = [(o.get("data") or {}).get("result") for o in observations]
        if synth.stable_share("tool-exchange", row.get("source_id")) < TOOL_EXCHANGE_RATE:
            try:
                turn = synth.exchange(str(row.get("user_prompt") or ""), clean_response(row.get("response")))
            except Reject:
                turn = None
            if turn:
                exchanges.append(turn)
    return results, exchanges


def collect_chat_exchanges(root: Path, limit: int | None) -> list[Any]:
    exchanges = []
    for index, row in enumerate(read_parquet_rows(chat_paths(root)["dpo"], limit)):
        chat = synth.dpo_chat(row)
        if chat and synth.stable_share("chat-exchange", index) < CHAT_EXCHANGE_RATE:
            try:
                turn = synth.exchange(chat[0], clean_response(chat[1]))
            except Reject:
                turn = None
            if turn:
                exchanges.append(turn)
    return exchanges


def collect_new_file_writes(root: Path, limit: int | None) -> set[tuple[str, int]]:
    """Hunter TR (source_id, step) whose write_file is a FILE_CREATE (§2.1)."""
    arguments = {}
    for row in read_rows(sources(root, "yurutme")[-1][1], limit):
        arguments[(str(row.get("source_id")), int(row.get("step_index") or 0))] = (
            row.get("action"), row.get("target_arguments") or {},
        )
    episodes = (
        (str(row.get("source_id")), row.get("observations") or [])
        for row in read_rows(sources(root, "yargi")[-1][1], limit)
    )
    return new_file_write_steps(arguments, episodes)


# ------------------------------------------------------------- records

class Context:
    def __init__(self, aliases, queries, calc_results, new_writes, history_pool=()):
        self.aliases = aliases
        self.queries = queries
        self.calc_results = calc_results
        self.new_writes = new_writes
        self.history_pool = list(history_pool)

    def query_lookup(self, source_id: str):
        return lambda step: self.queries.get((source_id, step))

    def step_aliases(self, source_id: str):
        return lambda step: WRITE_AS_CREATE if (source_id, step) in self.new_writes else {}

    def action(self, name, source_id: str, step: int):
        return canonical_action(name, {**self.aliases, **self.step_aliases(source_id)(step)})

    def rough_label(self, role: str, row: dict[str, Any]) -> str | None:
        """Label used only to choose history-mixing rates."""
        source_id, step = str(row.get("source_id")), int(row.get("step_index") or 0)
        if role == "yasama":
            return self.action(row.get("target_action"), source_id, step)
        if role == "yurutme":
            return (self.action(row.get("mapped_action"), source_id, step)
                    or self.action(row.get("action"), source_id, step) or "OPEN_TOOL")
        actions = [self.action(o.get("action"), source_id, i) or "OPEN_TOOL"
                   for i, o in enumerate(row.get("observations") or []) if isinstance(o, dict)]
        return actions[0] if actions else None


def yasama_record(row: dict[str, Any], source: str, ctx: Context):
    source_id = str(row.get("source_id"))
    step = int(row.get("step_index") or 0)
    action = ctx.action(row.get("target_action"), source_id, step)
    if action is None:
        raise Reject("unmapped_action")
    state = normalize_state(
        row.get("state"), ctx.aliases, step_index=step,
        turn_index=int(row.get("turn_index") or 0), query_for_step=ctx.query_lookup(source_id),
        step_aliases=ctx.step_aliases(source_id),
    )
    if action == "FINISH" and not state["recent_observations"]:
        raise Reject("finish_without_observation")
    history = normalize_history(row.get("conversation_history"))
    messages = [
        yasama_request_message(user_prompt=row["user_prompt"], conversation_history=history, state=state),
        {"role": "assistant", "content": action},
    ]
    check_record_length(messages)
    return action, messages, {"step_index": step, "has_history": bool(history)}


def yurutme_record(row: dict[str, Any], source: str, ctx: Context):
    """Registry tools as before; any other tool keeps its source name and its
    own schema (§2.2 open tools): Yurutme fills whatever schema it is given."""
    source_id = str(row.get("source_id"))
    step = int(row.get("step_index") or 0)
    action = ctx.action(row.get("mapped_action"), source_id, step) or ctx.action(row.get("action"), source_id, step)
    if action in CONTROL_ACTIONS:
        raise Reject("not_a_tool_action")
    meta: dict[str, Any] = {"step_index": step}
    if action is None:
        name = row.get("action")
        if not isinstance(name, str) or not name:
            raise Reject("unmapped_action")
        tool_schema = open_tool_schema(row.get("tool_schema"), row.get("placeholder"))
        arguments = validate_open_arguments(name, row.get("target_arguments"), tool_schema)
        action, label, placeholder = name, "OPEN_TOOL", tool_schema["placeholder"]
        meta["tool"] = name
    else:
        arguments = normalize_arguments(action, row.get("target_arguments"))
        if action == "CALCULATE":
            results = ctx.calc_results.get(source_id)
            if not results or step >= len(results):
                raise Reject("calculate_unverified")
            source_arguments = row.get("target_arguments") or {}
            check_calculation(arguments["expression"], results[step], row["user_prompt"],
                              operands_from_prompt="operation" in source_arguments)
        tool = BUILTIN_TOOL_REGISTRY[action]
        tool_schema, label, placeholder = public_tool_schema(tool), action, tool.get("placeholder")
    state = normalize_state(
        row.get("state"), ctx.aliases, step_index=step,
        turn_index=int(row.get("turn_index") or 0), query_for_step=ctx.query_lookup(source_id),
        step_aliases=ctx.step_aliases(source_id), keep_unmapped=True,
    )
    history = normalize_history(row.get("conversation_history"))
    messages = [
        yurutme_request_message(
            user_prompt=row["user_prompt"], conversation_history=history, state=state, action=action,
            tool_schema=tool_schema, placeholder=placeholder,
        ),
        {"role": "assistant", "content": json.dumps(arguments, ensure_ascii=False)},
    ]
    check_record_length(messages)
    meta["has_history"] = bool(history)
    return label, messages, meta


def yargi_record(row: dict[str, Any], source: str, ctx: Context):
    raw = row.get("observations") or ([row["observation"]] if isinstance(row.get("observation"), dict) else [])
    if not raw and not row.get("respond"):
        # atasoglu: answers without tool results (§7). RESPOND turns have none by design.
        raise Reject("empty_observations")
    source_id = str(row.get("source_id"))
    observations = [
        normalize_observation(o, ctx.aliases, query=ctx.queries.get((source_id, index)),
                              extra_aliases=ctx.step_aliases(source_id)(index), keep_unmapped=True)
        for index, o in enumerate(raw)
    ]
    response = clean_response(row.get("response"))
    check_answer_form(response)
    results = [
        o["data"]["result"] for o in observations
        if o["action"] == "CALCULATE" and o["status"] == "success"
        and isinstance(o["data"].get("result"), (int, float)) and not isinstance(o["data"]["result"], bool)
    ]
    if results:
        check_numbers_against_results(response, results, row["user_prompt"])
    history = normalize_history(row.get("conversation_history"))
    messages = [
        *yargi_request_messages(user_prompt=row["user_prompt"], conversation_history=history, observations=observations),
        {"role": "assistant", "content": response},
    ]
    check_record_length(messages)
    registry = set(BUILTIN_TOOL_REGISTRY)
    label = "+".join(sorted({o["action"] if o["action"] in registry else "OPEN_TOOL" for o in observations})) or "RESPOND"
    errors = sum(o["status"] == "error" for o in observations)
    return label, messages, {"has_history": bool(history), "error_observations": errors}


BUILDERS = {"yasama": yasama_record, "yurutme": yurutme_record, "yargi": yargi_record}


def history_mix_rows(root: Path, role: str, ctx: Context, limit: int | None) -> Iterator[dict[str, Any]]:
    """Existing records with an unrelated earlier exchange (synth module doc).
    Yasama and Yurutme take first-step rows (Yasama sees history only there);
    the copy keeps the original's split so no test episode leaks into train."""
    if not ctx.history_pool:
        return
    for source, path in sources(root, role):
        for row in read_rows(path, limit):
            if row.get("conversation_history") or (role != "yargi" and int(row.get("step_index") or 0) != 0):
                continue
            label = ctx.rough_label(role, row)
            key = (source, row.get("source_id"), row.get("step_index"))
            if label is None or synth.stable_share("hist", *key) >= synth.history_rate(label):
                continue
            mixed = synth.with_history(row, synth.pick(ctx.history_pool, *key))
            mixed["_split"] = split_for(source, row)
            yield mixed


def chat_rows(root: Path, role: str, limit: int | None) -> Iterator[dict[str, Any]]:
    paths = chat_paths(root)
    return synth.chat_respond_rows(
        role,
        read_parquet_rows(paths["dpo"], limit),
        read_parquet_rows(paths["human_like"], limit),
        read_parquet_rows(paths["helpsteer"], limit),
        read_parquet_rows(paths["helpsteer_holdout"], limit),
        dpo_rate=DPO_RESPOND_RATE,
    )


def role_inputs(root: Path, role: str, ctx: Context, limit: int | None):
    """(source name, description, rows) in build order."""
    for source, path in sources(root, role):
        yield source, path.name, read_rows(path, limit)
    if role != "yurutme":
        yield "chat_respond", "dpo + Human-Like + HelpSteer3", chat_rows(root, role, limit)
    calc = list(short_calc.episodes(limit or short_calc.EPISODES))
    # Yurutme verifies a CALCULATE row against the episode's recorded results.
    ctx.calc_results.update({item["source_id"]: [item["result"]] for item in calc})
    yield short_calc.SOURCE, "templated everyday calculations", short_calc.rows(role, calc)
    yield "history_mix", "existing records + earlier turn", history_mix_rows(root, role, ctx, limit)


# ---------------------------------------------------------------- build

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(root: Path, out: Path, *, version: str, limit: int | None, hash_sources: bool,
          revision: str | None = None) -> dict[str, Any]:
    started = time.time()
    mapping_path = root / "yasama-yurutme-yargi" / "ontology" / "YASAMA_16CLASS_MAPPING_V1.json"
    aliases = load_aliases(json.loads(mapping_path.read_text(encoding="utf-8")))
    log(f"aliases: {len(aliases)}")
    queries = collect_web_queries(root, limit)
    log(f"web queries: {len(queries)}")
    calc_results, tool_exchanges = collect_calc_results(root, limit)
    log(f"calculator episodes: {len(calc_results)}")
    chat_exchanges = collect_chat_exchanges(root, limit)
    log(f"history pool: {len(tool_exchanges)} tool turns + {len(chat_exchanges)} chat turns")
    new_writes = collect_new_file_writes(root, limit)
    log(f"write_file as FILE_CREATE: {len(new_writes)}")
    ctx = Context(aliases, queries, calc_results, new_writes, tool_exchanges + chat_exchanges)

    if out.exists():
        raise SystemExit(f"{out} exists; pick a new --version or remove it")
    (out / "contracts").mkdir(parents=True)
    (out / "rejects").mkdir()
    shutil.copyfile(REGISTRY_PATH, out / "contracts" / "tool_registry.json")

    counts: dict[str, Any] = {}
    rejects: dict[str, Any] = {}
    for role in ROLES:
        (out / role).mkdir()
        writers = {split: gzip.open(out / role / f"{split}.jsonl.gz", "wt", encoding="utf-8") for split in SPLITS}
        seen: set[bytes] = set()
        open_tools: set[str] = set()
        role_counts = defaultdict(lambda: defaultdict(Counter))      # split -> action -> source
        role_rejects = defaultdict(Counter)                           # source -> reason
        extra = Counter()
        samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        try:
            for source, description, rows in role_inputs(root, role, ctx, limit):
                for index, row in enumerate(rows):
                    try:
                        label, messages, meta = BUILDERS[role](row, source, ctx)
                    except Reject as exc:
                        role_rejects[source][exc.reason] += 1
                        if len(samples[exc.reason]) < REJECT_SAMPLES:
                            samples[exc.reason].append({
                                "source": source, "source_id": row.get("source_id"),
                                "row": json.dumps(row, ensure_ascii=False)[:1500],
                            })
                        continue
                    key = hashlib.sha1(json.dumps(messages, ensure_ascii=False).encode()).digest()
                    if key in seen:
                        role_rejects[source]["duplicate"] += 1
                        continue
                    seen.add(key)
                    split = row.get("_split") or split_for(source, row)
                    record = {
                        "id": f"{source}:{row.get('source_id')}:{meta.get('step_index', 0)}:{index}",
                        "source": source,
                        "source_dataset": row.get("source_dataset"),
                        "split": split,
                        "action": label,
                        **meta,
                        "messages": messages,
                    }
                    writers[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                    role_counts[split][label][source] += 1
                    extra["history"] += meta.get("has_history", False)
                    extra["error_observations"] += bool(meta.get("error_observations"))
                    if meta.get("tool"):
                        open_tools.add(meta["tool"])
                log(f"{role} {source} {description}: done")
        finally:
            for writer in writers.values():
                writer.close()
        counts[role] = {
            "splits": {s: sum(sum(c.values()) for c in role_counts[s].values()) for s in SPLITS},
            "by_action": {
                label: {s: sum(role_counts[s][label].values()) for s in SPLITS}
                for label in sorted({l for s in SPLITS for l in role_counts[s]})
            },
            "by_action_source": {
                label: dict(sum((role_counts[s][label] for s in SPLITS), Counter()))
                for label in sorted({l for s in SPLITS for l in role_counts[s]})
            },
            "with_history": extra["history"],
            "with_error_observation": extra["error_observations"],
            "distinct_open_tools": len(open_tools),
        }
        rejects[role] = {source: dict(c.most_common()) for source, c in role_rejects.items()}
        with (out / "rejects" / f"{role}.samples.jsonl").open("w", encoding="utf-8") as handle:
            for reason, rows in samples.items():
                for sample in rows:
                    handle.write(json.dumps({"reason": reason, **sample}, ensure_ascii=False) + "\n")

    source_files = sorted({str(p) for role in ROLES for _, p in sources(root, role)}
                          | {str(p) for paths in chat_paths(root).values() for p in paths})
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_version": version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "contract": f"PROJECT_MASTER {CONTRACT_VERSION}",
        "ontology": "contracts/tool_registry.json (17 actions)",
        "serialization": {
            "code_revision": revision or code_revision(),
            "builders": "yasama.request_message, yurutme.request_message, yargi.request_messages",
            "system_prompt": "not included; chosen at training time (§14)",
        },
        "sources": {
            path: {"sha256": sha256(Path(path)) if hash_sources else None} for path in source_files
        },
        "split_policy": "combined_v2 keeps its split; others by md5(source:source_id): test 2%, validation 2%",
        "dedup_policy": "exact duplicate messages per role dropped (first kept)",
        "limit_per_file": limit,
        "counts": counts,
        "rejects": rejects,
        "verifier": "build-time: every record passed normalize + registry validation; see rejects",
        "build_seconds": round(time.time() - started),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


def code_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, type=Path, help="tooluse_inspect directory")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--limit", type=int, help="rows per source file (smoke runs)")
    parser.add_argument("--no-hash", action="store_true", help="skip source sha256")
    parser.add_argument("--code-revision", help="SoP commit, when running from an archive without .git")
    args = parser.parse_args(argv)
    manifest = build(args.root, args.out, version=args.version, limit=args.limit,
                     hash_sources=not args.no_hash, revision=args.code_revision)
    print(json.dumps(manifest["counts"], ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
