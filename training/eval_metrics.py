"""Scores for generated role outputs (training/eval_role.py). Pure Python:
each function takes the request text, the reference target and the model
output, so the checks run on TRUBA and in the unit tests alike."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from datatools.normalize import is_english
from runtime.local_tools import _calculate
from runtime.output_parsing import parse_json_object
from runtime.tool_validation import validate_tool_arguments

# ---------------------------------------------------------------- Yasama


def score_yasama(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Accuracy, per-action recall / precision and the most common confusions."""
    total = correct = 0
    support: Counter = Counter()
    hits: Counter = Counter()
    predicted: Counter = Counter()
    confusions: Counter = Counter()
    for record in records:
        gold, output = record["target"].strip(), record["output"].strip()
        total += 1
        support[gold] += 1
        predicted[output] += 1
        if output == gold:
            correct += 1
            hits[gold] += 1
        else:
            confusions[f"{gold} -> {output or '<empty>'}"] += 1
    return {
        "records": total,
        "accuracy": round(correct / max(1, total), 4),
        "recall": {a: round(hits[a] / n, 4) for a, n in support.most_common()},
        "precision": {a: round(hits[a] / n, 4) for a, n in predicted.most_common() if a in support},
        "invalid_outputs": sum(n for a, n in predicted.items() if a not in support),
        "top_confusions": dict(confusions.most_common(20)),
    }


# --------------------------------------------------------------- Yurutme

_SCHEMA = re.compile(r"\nTOOL_SCHEMA:\n(.*)\nPLACEHOLDER:\n", re.S)
_ACTION = re.compile(r"\nACTION:\n(.*?)\n")


def _normal(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_normal(v) for v in value]
    if isinstance(value, dict):
        return {k: _normal(v) for k, v in value.items()}
    return value


def _calculation(expression: Any) -> Any:
    observation = _calculate({"expression": expression})
    return None if observation["status"] != "success" else observation["data"]["result"]


def _same_number(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b and a is not None
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    return abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))


def score_yurutme_one(request: str, target: str, output: str) -> dict[str, Any]:
    """One record: is the output a JSON object, does it pass the request's own
    tool schema (as the runtime validates it), does it equal the reference,
    and which share of the reference fields it reproduces. CALCULATE counts
    as correct when both expressions evaluate to the same number."""
    action = (_ACTION.search(request) or [None, "?"])[1]
    schema_match = _SCHEMA.search(request)
    schema = json.loads(schema_match.group(1)) if schema_match else {}
    gold = json.loads(target)
    result = {"action": action, "json": False, "schema_valid": False, "exact": False,
              "semantic": False, "field_match": 0.0}
    try:
        arguments = parse_json_object(output)
    except (ValueError, TypeError):
        return result
    result["json"] = True
    try:
        validate_tool_arguments(action, arguments, schema)
        result["schema_valid"] = True
    except (ValueError, TypeError):
        pass
    result["exact"] = _normal(arguments) == _normal(gold)
    if gold:
        result["field_match"] = round(sum(_normal(arguments.get(k)) == _normal(v) for k, v in gold.items()) / len(gold), 4)
    if action == "CALCULATE" and "expression" in gold:
        result["semantic"] = result["schema_valid"] and _same_number(
            _calculation(arguments.get("expression")), _calculation(gold["expression"]))
    else:
        result["semantic"] = result["schema_valid"] and result["exact"]
    if action in {"CODE_EXECUTE", "FILE_CREATE"}:
        # Long contents never match verbatim; the target path and that Python
        # code parses are what can be checked.
        code = arguments.get("code") if action == "CODE_EXECUTE" else arguments.get("content")
        is_python = action == "CODE_EXECUTE" and arguments.get("language", "python") == "python" or \
            str(arguments.get("path", "")).endswith(".py")
        result["python_parses"] = _parses(code) if is_python and isinstance(code, str) else None
        if action == "FILE_CREATE":
            result["path_match"] = _normal(arguments.get("path")) == _normal(gold.get("path"))
    return result


def _parses(code: str) -> bool:
    try:
        compile(code, "<eval>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def score_yurutme(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_action[record["action"]].append(score_yurutme_one(record["request"], record["target"], record["output"]))
    report: dict[str, Any] = {}
    every = [r for rows in by_action.values() for r in rows]
    for name, rows in [("ALL", every), *sorted(by_action.items(), key=lambda kv: -len(kv[1]))]:
        entry = {"records": len(rows)}
        for key in ("json", "schema_valid", "exact", "semantic"):
            entry[key] = round(sum(r[key] for r in rows) / max(1, len(rows)), 4)
        entry["field_match"] = round(sum(r["field_match"] for r in rows) / max(1, len(rows)), 4)
        for key in ("python_parses", "path_match"):
            values = [r[key] for r in rows if r.get(key) is not None]
            if values:
                entry[key] = round(sum(values) / len(values), 4)
        report[name] = entry
    return report


# ------------------------------------------------------------------ Yargi

_OBSERVATIONS = re.compile(r"<tool_observations>\n(.*)\n</tool_observations>", re.S)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_FAILURE = re.compile(
    r"hata|başarısız|yapılamadı|bulunamadı|bulamadım|olmadı|maalesef|ne yazık ki|mümkün olmadı|"
    r"gerçekleştirilemedi|erişilemedi|okunamadı|oluşturulamadı|silinemedi|izin",
    re.IGNORECASE,
)


def _numbers(text: str) -> set[str]:
    """Number readings with '.'/',' as either separator, digits only."""
    readings = set()
    for match in _NUMBER.findall(text):
        readings.add(re.sub(r"[.,]", "", match))
        readings.add(match.replace(",", "."))
        readings.add(match.replace(".", "").replace(",", "."))
    return readings


def score_yargi_one(request: str, target: str, output: str) -> dict[str, Any]:
    """Faithfulness checks that need no judge: JSON leaks, English, empty
    answers, numbers absent from the request and observations, and whether
    an answer to a failed tool says it failed."""
    observations_text = (_OBSERVATIONS.search(request) or [None, ""])[1]
    try:
        observations = json.loads(observations_text) if observations_text else []
    except ValueError:
        observations = []
    has_error = any(isinstance(o, Mapping) and o.get("status") == "error" for o in observations)
    stripped = output.strip()
    source_numbers = _numbers(request)
    output_numbers = {m for m in _NUMBER.findall(stripped)}
    ungrounded = sorted(m for m in output_numbers
                        if not (_numbers(m) & source_numbers) and len(re.sub(r"\D", "", m)) > 1)
    words = lambda text: set(re.findall(r"[\wçğıöşü]+", text.lower()))  # noqa: E731
    gold_words, out_words = words(target), words(stripped)
    overlap = len(gold_words & out_words)
    f1 = 0.0 if not overlap else 2 * overlap / (len(gold_words) + len(out_words))
    return {
        "empty": not stripped,
        "json_leak": stripped.startswith("{") or '"action"' in stripped or '"status"' in stripped,
        "english": bool(stripped) and is_english(stripped),
        "ungrounded_numbers": ungrounded,
        "has_error_observation": has_error,
        "admits_failure": bool(_FAILURE.search(stripped)) if has_error else None,
        "no_observations": not observations,
        "word_f1": round(f1, 4),
        "length_ratio": round(len(stripped) / max(1, len(target)), 3),
    }


def score_yargi(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_action[record["action"]].append(score_yargi_one(record["request"], record["target"], record["output"]))
    report: dict[str, Any] = {}
    every = [r for rows in by_action.values() for r in rows]
    for name, rows in [("ALL", every), *sorted(by_action.items(), key=lambda kv: -len(kv[1]))]:
        if name != "ALL" and len(rows) < 10:
            continue
        n = max(1, len(rows))
        errors = [r for r in rows if r["has_error_observation"]]
        report[name] = {
            "records": len(rows),
            "empty": round(sum(r["empty"] for r in rows) / n, 4),
            "json_leak": round(sum(r["json_leak"] for r in rows) / n, 4),
            "english": round(sum(r["english"] for r in rows) / n, 4),
            "with_ungrounded_numbers": round(sum(bool(r["ungrounded_numbers"]) for r in rows
                                                 if not r["no_observations"]) / max(1, sum(not r["no_observations"] for r in rows)), 4),
            "error_records": len(errors),
            "admits_failure": round(sum(r["admits_failure"] for r in errors) / len(errors), 4) if errors else None,
            "word_f1": round(sum(r["word_f1"] for r in rows) / n, 4),
            "median_length_ratio": sorted(r["length_ratio"] for r in rows)[len(rows) // 2] if rows else None,
        }
    return report


SCORERS = {"yasama": score_yasama, "yurutme": score_yurutme, "yargi": score_yargi}
