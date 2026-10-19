"""Normalize source tool-use records to the PROJECT_MASTER contract.

Pure functions; ``build_shared_data`` streams the sources through them. Every
rule here is documented in PROJECT_MASTER §2.2 (argument names), §3.1
(observation shapes), §3.2 (state) and §6 (calculator). A record that cannot
be brought to the contract raises ``Reject`` with a short reason, which the
build counts instead of guessing.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from runtime.local_tools import BUILTIN_TOOL_REGISTRY, execute_builtin_local
from runtime.tool_validation import validate_tool_arguments
from yasama.yasama import ACTION_ORDER, CONTROL_ACTION_DESCRIPTIONS

ACTIONS = frozenset(ACTION_ORDER)
CONTROL_ACTIONS = frozenset(CONTROL_ACTION_DESCRIPTIONS)

# PROJECT_MASTER §2.1 safe aliases, plus calc/calculator (§6). The ontology
# file's mapped_safe_alias rows are added by load_aliases().
BASE_ALIASES = {
    "web_search": "WEB_SEARCH",
    "read_file": "FILE_READ",
    "edit_file": "FILE_EDIT",
    "list_directory": "FILE_LIST",
    "search_code": "FILE_SEARCH",
    "run_command": "SHELL_EXECUTE",
    "calc": "CALCULATE",
    "calculator": "CALCULATE",
}


class Reject(Exception):
    """The record cannot be brought to the contract; ``reason`` is counted."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def load_aliases(ontology_mapping: Mapping[str, Any] | None) -> dict[str, str]:
    aliases = dict(BASE_ALIASES)
    for row in (ontology_mapping or {}).get("mapping", []):
        if row.get("mapping_status") == "mapped_safe_alias" and row.get("ontology_action") in ACTIONS:
            aliases[row["source_action"]] = row["ontology_action"]
    return aliases


def canonical_action(name: Any, aliases: Mapping[str, str]) -> str | None:
    if not isinstance(name, str):
        return None
    if name in ACTIONS:
        return name
    return aliases.get(name)


# --------------------------------------------------------------- arguments

ARGUMENT_RENAMES: dict[str, dict[str, str]] = {
    "FILE_LIST": {"dir_path": "path", "directory": "path"},
    "FILE_READ": {"file_path": "path"},
    "FILE_EDIT": {"file_path": "path"},
    "FILE_SEARCH": {"file_pattern": "pattern", "directory": "path"},
    "FILE_MOVE": {"source_path": "source", "destination_path": "destination"},
    "SEND_MAIL": {"recipient": "to", "message": "body"},
    "DOWNLOAD": {"output_path": "path", "destination": "path", "target_directory": "path"},
    "WEB_SEARCH": {"q": "query", "num": "max_results", "limit": "max_results"},
}
# Fields with no counterpart in the registry. FILE_READ limit/offset paged
# a file; long files are now read through retrieval (§3.1).
ARGUMENT_DROPS: dict[str, frozenset[str]] = {
    "FILE_READ": frozenset({"limit", "offset"}),
    "SHELL_EXECUTE": frozenset({"timeout"}),
    "WEB_SEARCH": frozenset({"related_keywords", "country", "lang", "gl", "lr", "start"}),
}
CALCULATOR_OPERATORS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
MAX_WEB_RESULTS = 3


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _operand(value: Any) -> str:
    text = repr(value) if isinstance(value, float) else str(value)
    return f"({text})" if value < 0 else text


def calculator_expression(arguments: Mapping[str, Any]) -> str:
    """calculator {operation, a, b} -> CALCULATE expression (§6)."""
    operator = CALCULATOR_OPERATORS.get(arguments.get("operation"))
    a, b = arguments.get("a"), arguments.get("b")
    if operator is None or not _is_number(a) or not _is_number(b):
        raise Reject("calculator_unconvertible")
    return f"{_operand(a)}{operator}{_operand(b)}"


def normalize_arguments(action: str, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, Mapping):
        raise Reject("arguments_not_object")
    args = dict(arguments)
    if action == "CALCULATE" and set(args) == {"operation", "a", "b"}:
        args = {"expression": calculator_expression(args)}
    if action == "FILE_SEARCH" and "query" not in args:
        # Gemini distil keeps the searched text in `pattern` (§2.2).
        if "pattern" in args:
            args["query"] = args.pop("pattern")
        if "extension" in args:
            args["pattern"] = "*." + str(args.pop("extension")).lstrip("*.")
    if action == "CODE_EXECUTE":
        if args.pop("answer", None) is not None:
            raise Reject("code_execute_answer_in_arguments")
        args.setdefault("language", "python")
    if action == "FILE_DELETE" and "paths" in args:
        paths = args.pop("paths")
        if not (isinstance(paths, list) and len(paths) == 1):
            raise Reject("file_delete_many_paths")
        args["path"] = paths[0]
    for old, new in ARGUMENT_RENAMES.get(action, {}).items():
        if old in args:
            if new in args:
                raise Reject(f"argument_conflict:{old}")
            args[new] = args.pop(old)
    for key in ARGUMENT_DROPS.get(action, ()):
        args.pop(key, None)
    if action == "WEB_SEARCH" and "max_results" in args:
        try:
            args["max_results"] = max(1, min(int(args["max_results"]), MAX_WEB_RESULTS))
        except (TypeError, ValueError):
            args.pop("max_results")
    tool = BUILTIN_TOOL_REGISTRY.get(action)
    if tool is None:
        raise Reject("not_a_tool_action")
    try:
        return validate_tool_arguments(action, args, tool)
    except ValueError as exc:
        raise Reject("schema:" + _short(str(exc))) from exc


def _short(message: str) -> str:
    return re.sub(r"[\"'].*?[\"']", "…", message)[:80]


# ------------------------------------------------------------ observations

_WEB_BLOCK = re.compile(r"\[(\d+)\]\s+(\S+)\n(.*?)(?=\n+\[\d+\]\s+\S+\n|\Z)", re.S)
_SIZE_SUFFIX = re.compile(r"\s+\(\d+ bytes\)$")


def parse_web_result_text(text: str) -> list[dict[str, str]]:
    """``"[1] url\\nsnippet\\n\\n[2] url\\n..."`` -> [{url, snippet}] (§6)."""
    results = [
        {"url": url, "snippet": snippet.strip()}
        for _, url, snippet in _WEB_BLOCK.findall(text.strip())
    ]
    if not results:
        raise Reject("web_result_unparsed")
    return results[:MAX_WEB_RESULTS]


def _first(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def normalize_data(action: str, data: Any, *, query: str | None = None) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        data = {"result": data}
    data = dict(data)
    if action == "WEB_SEARCH":
        results = data.get("results")
        if not isinstance(results, list):
            text = data.get("result")
            if not isinstance(text, str):
                raise Reject("web_result_missing")
            results = parse_web_result_text(text)
        shaped: dict[str, Any] = {"query": query} if query else {}
        shaped["results"] = results[:MAX_WEB_RESULTS]
        return shaped
    if action == "FILE_LIST":
        entries = _first(data, "entries", "files", "items", "result")
        if not isinstance(entries, list):
            raise Reject("file_list_entries_missing")
        entries = [_SIZE_SUFFIX.sub("", str(e)) for e in entries]
        entries += [str(d).rstrip("/") + "/" for d in data.get("directories") or []]
        return {**({"path": data["path"]} if "path" in data else {}), "entries": entries}
    if action == "FILE_READ":
        content = _first(data, "content", "text", "result")
        if not isinstance(content, str):
            raise Reject("file_read_content_missing")
        return {**({"path": data["path"]} if "path" in data else {}), "content": content}
    if action == "FILE_SEARCH":
        matches = _first(data, "matches", "results", "found_files", "files", "result")
        if not isinstance(matches, list):
            raise Reject("file_search_matches_missing")
        matches = [m if isinstance(m, Mapping) else {"path": str(m)} for m in matches]
        return {**({"query": data["query"]} if "query" in data else {}), "matches": matches}
    if action == "WEB_OPEN":
        text = _first(data, "text", "content")
        if not isinstance(text, str):
            raise Reject("web_open_text_missing")
        return {key: data[key] for key in ("url", "title") if key in data} | {"text": text}
    if action == "CALCULATE":
        if not _is_number(data.get("result")) and not isinstance(data.get("result"), bool):
            raise Reject("calculate_result_missing")
        return {"result": data["result"]}
    return data


def normalize_observation(
    observation: Any, aliases: Mapping[str, str], *, query: str | None = None
) -> dict[str, Any]:
    if not isinstance(observation, Mapping):
        raise Reject("observation_not_object")
    action = canonical_action(observation.get("action"), aliases)
    if action is None:
        raise Reject("observation_unmapped_action")
    if action in CONTROL_ACTIONS:
        # Gemini distil: the target answer stored as a RESPOND observation.
        raise Reject("observation_control_action")
    data = observation.get("data")
    status = observation.get("status") or "success"
    error = observation.get("error")
    if isinstance(data, Mapping) and {"success", "result", "tool"} <= set(data):
        # Hunter TR wraps every tool result.
        if data.get("success") is False:
            status, error = "error", error or str(data.get("result") or "error")
        data = data.get("result")
    if status not in {"success", "error"}:
        raise Reject("observation_bad_status")
    if status == "error":
        return {"action": action, "data": {}, "status": "error", "error": str(error or "error")}
    return {
        "action": action,
        "data": normalize_data(action, data, query=query),
        "status": "success",
        "error": None,
    }


QueryLookup = Callable[[int], "str | None"]


def normalize_state(
    state: Any,
    aliases: Mapping[str, str],
    *,
    step_index: int,
    turn_index: int,
    query_for_step: QueryLookup = lambda step: None,
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        state = {}
    completed = []
    for item in state.get("completed_actions") or []:
        name = item.get("action") if isinstance(item, Mapping) else item
        action = canonical_action(name, aliases)
        if action is None:
            raise Reject("state_unmapped_action")
        completed.append(action)
    recent_source = state.get("recent_observations")
    last_source = state.get("last_observation")
    if recent_source is None:
        # Older sources keep only the last observation.
        recent_source = [last_source] if last_source else []
    recent = [
        normalize_observation(obs, aliases, query=query_for_step(index))
        for index, obs in enumerate(recent_source)
    ]
    last = None
    if last_source:
        last = normalize_observation(last_source, aliases, query=query_for_step(len(recent_source) - 1))
    return {
        "completed_actions": completed,
        "last_observation": last,
        "recent_observations": recent,
        "step_index": step_index,
        "turn_index": turn_index,
    }


def normalize_history(history: Any) -> list[dict[str, str]]:
    return [
        {"role": m["role"], "content": m["content"]}
        for m in history or []
        if isinstance(m, Mapping) and m.get("role") in {"user", "assistant"}
        and isinstance(m.get("content"), str)
    ]


# ------------------------------------------------------------- responses

_TR_WORDS = frozenset("ve bir bu için ile de da olarak dosya şu çok daha göre veya ancak sonuç bulunan".split())
_EN_WORDS = frozenset("the and is to of you your i've i'm this that with for are was it".split())
_ANSWER_TAG = re.compile(r"</?answer>")
_BOXED_TAIL = re.compile(
    r"\s*(?:(?:Bu nedenle|Bu yüzden|Dolayısıyla|Yani|Sonuç olarak),?\s*)?\\?boxed\s*\{?\s*[^\s{}]*\s*\}?\s*\.?\s*$"
)
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def is_english(text: str) -> bool:
    lowered = text.lower()
    words = re.findall(r"[\wçğıöşü']+", lowered)
    tr = sum(w in _TR_WORDS for w in words) + (2 if re.search(r"[çğıöşü]", lowered) else 0)
    en = sum(w in _EN_WORDS for w in words)
    return en >= 3 and en > 1.5 * max(tr, 1)


def clean_response(response: Any) -> str:
    if not isinstance(response, str) or not response.strip():
        raise Reject("response_empty")
    text = _ANSWER_TAG.sub("", response)          # ituperceptron wraps answers
    text = _BOXED_TAIL.sub("", text).strip()      # tool-math LaTeX leftovers
    if "boxed" in text:
        raise Reject("response_boxed")
    if not text:
        raise Reject("response_empty")
    if is_english(text):
        raise Reject("response_english")
    return text


def _significant_digits(token: str) -> int:
    return len(token.lstrip("-").replace(".", "").replace(",", "").lstrip("0"))


def check_numbers_against_results(response: str, results: Sequence[float], user_prompt: str) -> None:
    """Long numbers in a calculator answer must be a rounding of a result
    (§6: "1.1833976833976836" written as "1.183397668339766836" is rejected)."""
    prompt_tokens = set(_NUMBER.findall(user_prompt))
    for token in _NUMBER.findall(response):
        if _significant_digits(token) < 5 or token in prompt_tokens:
            continue
        if not any(abs(value - float(r)) <= tolerance
                   for value, tolerance in _readings(token) for r in results):
            raise Reject("response_number_mismatch")


def _readings(token: str) -> list[tuple[float, float]]:
    """(value, tolerance) for each way a number may be written: "0,91" and
    "0.91" as decimals, "960.000" and "23,248" also as thousands groups."""
    decimals = len(re.split(r"[.,]", token)[1]) if re.search(r"[.,]", token) else 0
    readings = [(float(token.replace(",", ".")), 0.5 * 10 ** (-decimals) + 1e-12)]
    if re.fullmatch(r"-?\d{1,3}(?:[.,]\d{3})+", token):
        readings.append((float(re.sub(r"[.,]", "", token)), 0.5))
    return readings


# ------------------------------------------------------------- calculator

_DIGIT_PROMPT = re.compile(r"[\d.,\s+\-*/x×÷=?()]+")


def check_calculation(
    expression: str, expected: Any, user_prompt: str, *, operands_from_prompt: bool = False
) -> None:
    """Recompute with the runtime CALCULATE; the source result must agree.
    For calculator {operation, a, b} rows (``operands_from_prompt``) a prompt
    written only in digits must contain both operands (§6); tool-math
    expressions legitimately use intermediate results."""
    if operands_from_prompt and _DIGIT_PROMPT.fullmatch(user_prompt.strip()):
        prompt_numbers = {abs(float(n.replace(",", "."))) for n in _NUMBER.findall(user_prompt)}
        for operand in re.findall(r"\d+(?:\.\d+)?(?:e[+-]?\d+)?", expression):
            if abs(float(operand)) not in prompt_numbers:
                raise Reject("calculate_operand_not_in_prompt")
    observation = execute_builtin_local("CALCULATE", {"expression": expression})
    if observation["status"] != "success":
        raise Reject("calculate_expression_fails")
    got = observation["data"]["result"]
    if isinstance(got, bool) or not _is_number(expected):
        raise Reject("calculate_result_not_number")
    if not math.isclose(float(got), float(expected), rel_tol=1e-9, abs_tol=1e-9):
        raise Reject("calculate_result_mismatch")
