"""Command-line interface for the Separation of Powers runtime."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from main import AgentPipeline
from runtime.speech import EMALightningSpeech
from runtime.mail_clarification import (
    is_mail_cancel_request,
    is_mail_send_request,
    label_mail_clarification_answer,
)
from runtime import LibMercanBackend, LlamaCppBackend, MercanCliBackend, OllamaBackend
from yasama.yasama import YasamaRuntime
from yargi.yargi import YargiRuntime
from yurutme.yurutme import YurutmeRuntime


DEFAULT_CONFIG: dict[str, Any] = {
    "roles": {
        "yasama": {
            "backend": "ollama",
            "model": "mercan-yasama",
            "base_url": "http://127.0.0.1:11434",
        },
        "yurutme": {
            "backend": "ollama",
            "model": "mercan-yurutme",
            "base_url": "http://127.0.0.1:11434",
        },
        "yargi": {
            "backend": "ollama",
            "model": "mercan-yargi",
            "base_url": "http://127.0.0.1:11434",
        },
    },
    "pipeline": {
        "max_steps": 12,
        "tool_registry": None,
        "executor": None,
        "debug": False,
        "speech": False,
    },
}


BACKEND_NAMES = ("ollama", "llamacpp", "mercan-cli", "libmercan")
ROLE_NAMES = ("yasama", "yurutme", "yargi")


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path!s} must contain a JSON object.")
    return data


def _parse_scalar(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _apply_set(config: dict[str, Any], assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"--set expects key=value, got {assignment!r}.")
    dotted_key, raw_value = assignment.split("=", 1)
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        raise ValueError("--set key cannot be empty.")

    cursor: dict[str, Any] = config
    for part in parts[:-1]:
        current = cursor.get(part)
        if current is None:
            current = {}
            cursor[part] = current
        if not isinstance(current, dict):
            raise ValueError(f"Cannot set nested key below non-object {part!r}.")
        cursor = current
    cursor[parts[-1]] = _parse_scalar(raw_value)


def _resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if getattr(args, "config", None):
        config = _deep_merge(config, _load_json(args.config))

    for assignment in getattr(args, "sets", []) or []:
        _apply_set(config, assignment)

    for role in ROLE_NAMES:
        role_cfg = config.setdefault("roles", {}).setdefault(role, {})
        backend = getattr(args, f"{role}_backend", None)
        model = getattr(args, f"{role}_model", None)
        base_url = getattr(args, f"{role}_base_url", None)
        if backend:
            role_cfg["backend"] = backend
        if model:
            role_cfg["model"] = model
        if base_url:
            role_cfg["base_url"] = base_url

    if getattr(args, "max_steps", None) is not None:
        config.setdefault("pipeline", {})["max_steps"] = args.max_steps
    if getattr(args, "tool_registry", None):
        config.setdefault("pipeline", {})["tool_registry"] = args.tool_registry
    if getattr(args, "executor", None):
        config.setdefault("pipeline", {})["executor"] = args.executor
    if getattr(args, "debug", False):
        config.setdefault("pipeline", {})["debug"] = True
    if getattr(args, "speech", False):
        config.setdefault("pipeline", {})["speech"] = True

    return config


def _build_backend(role: str, config: Mapping[str, Any]):
    backend_name = str(config.get("backend", "")).strip().lower()
    model = config.get("model")
    if not model:
        raise ValueError(f"roles.{role}.model is required.")

    if backend_name == "ollama":
        return OllamaBackend(
            model=str(model),
            base_url=str(config.get("base_url", "http://127.0.0.1:11434")),
        )

    if backend_name == "llamacpp":
        return LlamaCppBackend(
            base_url=str(config.get("base_url", "http://127.0.0.1:8080")),
            model=str(model) if model is not None else None,
            api_key=str(config["api_key"]) if config.get("api_key") else None,
        )

    if backend_name == "mercan-cli":
        return MercanCliBackend(
            model=str(model),
            binary=str(config.get("binary", "mercan")),
            threads=_optional_int(config.get("threads")),
            gpu_layers=_optional_int(config.get("gpu_layers")),
            plugins=_string_list(config.get("plugins")),
        )

    if backend_name == "libmercan":
        return LibMercanBackend(
            model=str(model),
            library=str(config["library"]) if config.get("library") else None,
            gpu_layers=_optional_int(config.get("gpu_layers")),
            use_mmap=_optional_bool(config.get("use_mmap")),
            check_tensors=_optional_bool(config.get("check_tensors")),
            n_ctx=_optional_int(config.get("n_ctx")),
            n_batch=_optional_int(config.get("n_batch")),
            threads=_optional_int(config.get("threads")),
            threads_batch=_optional_int(config.get("threads_batch")),
            plugins=_string_list(config.get("plugins")),
            role_system=str(config.get("role_system", "sistem")),
            role_user=str(config.get("role_user", "kullanici")),
            role_assistant=str(config.get("role_assistant", "asistan")),
        )

    raise ValueError(
        f"Unknown backend {backend_name!r} for role {role}. "
        f"Choose one of: {', '.join(BACKEND_NAMES)}."
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}.")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError("Expected a string or list of strings.")


def _load_callable(spec: str):
    if ":" not in spec:
        raise ValueError("Executor must use module:function syntax.")
    module_name, attribute = spec.split(":", 1)
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    if not callable(value):
        raise TypeError(f"{spec!r} does not resolve to a callable.")
    return value


def _load_tool_registry(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    registry = _load_json(path)
    for action, schema in registry.items():
        if not isinstance(action, str) or not isinstance(schema, Mapping):
            raise ValueError("Tool registry must map action names to JSON objects.")
    return registry


def _missing_executor(action: str, arguments: dict[str, Any]) -> Mapping[str, Any]:
    raise RuntimeError(
        f"Yasama selected {action}, but no executor is configured. "
        "Pass --executor module:function and, if needed, --tool-registry registry.json."
    )


def _debug_trace(event: str, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(dict(payload), ensure_ascii=False, default=str)
    print(f"[trace:{event}] {rendered}", file=sys.stderr)


def build_runtime(config: Mapping[str, Any]) -> tuple[AgentPipeline, list[Any]]:
    roles = config.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("Config must contain a roles object.")

    backends = {role: _build_backend(role, roles.get(role, {})) for role in ROLE_NAMES}
    yasama = YasamaRuntime(backend=backends["yasama"])
    yurutme = YurutmeRuntime(backend=backends["yurutme"])
    yargi = YargiRuntime(backend=backends["yargi"])

    pipeline_cfg = config.get("pipeline", {})
    if not isinstance(pipeline_cfg, Mapping):
        raise ValueError("pipeline config must be a JSON object.")

    tool_registry = _load_tool_registry(
        str(pipeline_cfg["tool_registry"]) if pipeline_cfg.get("tool_registry") else None
    )
    executor = (
        _load_callable(str(pipeline_cfg["executor"]))
        if pipeline_cfg.get("executor")
        else _missing_executor
    )

    pipeline = AgentPipeline(
        yasama=yasama,
        yurutme=yurutme,
        yargi=yargi,
        tool_registry=tool_registry,
        executor=executor,
        max_steps=int(pipeline_cfg.get("max_steps", 12)),
        trace_handler=_debug_trace if bool(pipeline_cfg.get("debug", False)) else None,
    )
    return pipeline, list(backends.values())


def _close_backends(backends: list[Any]) -> None:
    seen: set[int] = set()
    for backend in backends:
        identity = id(backend)
        if identity in seen:
            continue
        seen.add(identity)
        close = getattr(backend, "close", None)
        if callable(close):
            close()


def _history_add(history: list[dict[str, str]], user: str, assistant: str) -> None:
    history.append({"role": "user", "content": user})
    history.append({"role": "assistant", "content": assistant})


def _speech_from_config(config: Mapping[str, Any]) -> EMALightningSpeech | None:
    pipeline = config.get("pipeline", {})
    if not isinstance(pipeline, Mapping) or not pipeline.get("speech", False):
        return None
    return EMALightningSpeech()


def _speak_answer(speaker: EMALightningSpeech | None, answer: str) -> None:
    if speaker is None:
        return
    try:
        speaker.speak(answer)
    except Exception as exc:
        # Audio problems should not turn a successful text response into a
        # failed agent turn or prevent subsequent chat messages.
        print(f"[speech:error] {exc}", file=sys.stderr)


def cmd_run(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    pipeline, backends = build_runtime(config)
    speaker = _speech_from_config(config)
    try:
        response = pipeline.run(args.prompt)
        print(response)
        _speak_answer(speaker, response)
        return 0
    finally:
        _close_backends(backends)


def cmd_chat(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    pipeline, backends = build_runtime(config)
    speaker = _speech_from_config(config)
    history: list[dict[str, str]] = []
    pending_mail_request: str | None = None
    pending_mail_question: str | None = None
    print(
        "SeparationOfPowers ready. /exit ile çık, /config ile aktif ayarı gör, "
        "/clear ile konuşma geçmişini temizle."
    )
    try:
        while True:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                continue
            if prompt in {"/exit", "/quit"}:
                break
            if prompt == "/config":
                print(json.dumps(config, ensure_ascii=False, indent=2))
                continue
            if prompt == "/clear":
                history.clear()
                pending_mail_request = None
                pending_mail_question = None
                print("Konuşma geçmişi temizlendi.")
                continue
            if is_mail_cancel_request(prompt):
                pending_mail_request = None
                pending_mail_question = None
                response = "Tamam, mail göndermeyeceğim."
                print(response)
                _history_add(history, prompt, response)
                _speak_answer(speaker, response)
                continue

            effective_prompt = prompt
            if pending_mail_request is not None:
                followup = label_mail_clarification_answer(
                    prompt, pending_mail_question or ""
                )
                effective_prompt = pending_mail_request + "\n" + followup

            try:
                response = pipeline.run(
                    effective_prompt, conversation_history=history
                )
            except Exception as exc:
                # A failed execution is not an unanswered clarification.
                if pipeline.last_action != "ASK_CLARIFICATION":
                    pending_mail_request = None
                    pending_mail_question = None
                print(f"[error] {exc}", file=sys.stderr)
                continue

            print(response)
            if (
                pipeline.last_action == "ASK_CLARIFICATION"
                and is_mail_send_request(effective_prompt)
            ):
                pending_mail_request = effective_prompt
                pending_mail_question = response
            else:
                pending_mail_request = None
                pending_mail_question = None
            _history_add(history, prompt, response)
            _speak_answer(speaker, response)
        return 0
    finally:
        _close_backends(backends)


def cmd_show_config(args: argparse.Namespace) -> int:
    print(json.dumps(_resolve_config(args), ensure_ascii=False, indent=2))
    return 0


def cmd_init_config(args: argparse.Namespace) -> int:
    destination = Path(args.path).expanduser()
    if destination.exists() and not args.force:
        raise FileExistsError(f"{destination} already exists; pass --force to overwrite.")
    destination.write_text(
        json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(destination)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    roles = config.get("roles", {})
    ok = True
    print("Runtime configuration:")
    for role in ROLE_NAMES:
        role_cfg = roles.get(role, {}) if isinstance(roles, Mapping) else {}
        backend = role_cfg.get("backend")
        model = role_cfg.get("model")
        print(f"  {role:8s} backend={backend!s:10s} model={model}")
        if backend not in BACKEND_NAMES or not model:
            ok = False

        if backend == "libmercan":
            model_path = Path(str(model)).expanduser()
            if not model_path.is_file():
                print(f"    ! model file not found: {model_path}")
                ok = False
            library = role_cfg.get("library")
            if library and not Path(str(library)).expanduser().exists():
                print(f"    ! libmercan not found: {library}")
                ok = False

    pipeline_cfg = config.get("pipeline", {})
    if isinstance(pipeline_cfg, Mapping):
        registry = pipeline_cfg.get("tool_registry")
        executor = pipeline_cfg.get("executor")
        print(f"  tool_registry={registry or '<none>'}")
        print(f"  executor={executor or '<none>'}")
        print(f"  debug={bool(pipeline_cfg.get('debug', False))}")
        print(f"  speech={bool(pipeline_cfg.get('speech', False))}")
    print("OK" if ok else "Configuration has issues.")
    return 0 if ok else 1


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config key, e.g. --set roles.yasama.threads=8",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--tool-registry", help="Tool registry JSON path")
    parser.add_argument("--executor", help="Executor callable as module:function")
    parser.add_argument("--debug", action="store_true", help="Print pipeline action/args/observation traces")
    parser.add_argument(
        "--speech",
        action="store_true",
        help="Speak the final answer aloud using EMA Lightning Turkish TTS",
    )

    for role in ROLE_NAMES:
        parser.add_argument(
            f"--{role}-backend",
            choices=BACKEND_NAMES,
            dest=f"{role}_backend",
        )
        parser.add_argument(f"--{role}-model", dest=f"{role}_model")
        parser.add_argument(f"--{role}-base-url", dest=f"{role}_base_url")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mercan-sop",
        description="Mercan Yasama/Yurutme/Yargi runtime launcher",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one user prompt")
    _add_runtime_options(run_parser)
    run_parser.add_argument("prompt")
    run_parser.set_defaults(func=cmd_run)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive terminal chat")
    _add_runtime_options(chat_parser)
    chat_parser.set_defaults(func=cmd_chat)

    show_parser = subparsers.add_parser("show-config", help="Print resolved runtime config")
    _add_runtime_options(show_parser)
    show_parser.set_defaults(func=cmd_show_config)

    doctor_parser = subparsers.add_parser("doctor", help="Check local runtime configuration")
    _add_runtime_options(doctor_parser)
    doctor_parser.set_defaults(func=cmd_doctor)

    init_parser = subparsers.add_parser("init-config", help="Create a starter config file")
    init_parser.add_argument("path", nargs="?", default="sop.json")
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(func=cmd_init_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
