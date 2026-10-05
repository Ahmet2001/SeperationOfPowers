"""Top-level mercan-sop CLI entrypoint.

This wrapper keeps the main pipeline CLI in ``cli.py`` and adds lightweight
commands that do not need to construct the full Yasama/Yurutme/Yargi pipeline.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import cli as pipeline_cli
from runtime import GenerationConfig


def build_model_test_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mercan-sop model-test",
        description="Test one model/backend directly without entering the agent pipeline.",
    )
    parser.add_argument("backend", choices=pipeline_cli.BACKEND_NAMES)
    parser.add_argument("model", help="Backend model name or local .mercan model path")
    parser.add_argument(
        "--prompt",
        default="Merhaba. Bu bir runtime bağlantı testidir. Kısa bir cevap ver.",
    )
    parser.add_argument(
        "--system",
        default="Bu bir model runtime bağlantı testidir. Kullanıcıya kısa ve net cevap ver.",
    )
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--binary", default="mercan")
    parser.add_argument("--library")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--threads-batch", type=int)
    parser.add_argument("--gpu-layers", type=int)
    parser.add_argument("--n-ctx", type=int)
    parser.add_argument("--n-batch", type=int)
    parser.add_argument("--plugin", action="append", default=[])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--repeat-last-n", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def _model_test_backend_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "backend": args.backend,
        "model": args.model,
    }

    if args.backend == "ollama":
        config["base_url"] = args.base_url or "http://127.0.0.1:11434"
    elif args.backend == "llamacpp":
        config["base_url"] = args.base_url or "http://127.0.0.1:8080"
        if args.api_key:
            config["api_key"] = args.api_key
    elif args.backend == "mercan-cli":
        config["binary"] = args.binary
        if args.threads is not None:
            config["threads"] = args.threads
        if args.gpu_layers is not None:
            config["gpu_layers"] = args.gpu_layers
        if args.plugin:
            config["plugins"] = list(args.plugin)
    elif args.backend == "libmercan":
        if args.library:
            config["library"] = args.library
        if args.threads is not None:
            config["threads"] = args.threads
        if args.threads_batch is not None:
            config["threads_batch"] = args.threads_batch
        if args.gpu_layers is not None:
            config["gpu_layers"] = args.gpu_layers
        if args.n_ctx is not None:
            config["n_ctx"] = args.n_ctx
        if args.n_batch is not None:
            config["n_batch"] = args.n_batch
        if args.plugin:
            config["plugins"] = list(args.plugin)

    return config


def cmd_model_test(argv: list[str]) -> int:
    parser = build_model_test_parser()
    args = parser.parse_args(argv)
    backend = pipeline_cli._build_backend(
        "model-test",
        _model_test_backend_config(args),
    )
    try:
        response = backend.generate(
            [
                {"role": "system", "content": args.system},
                {"role": "user", "content": args.prompt},
            ],
            config=GenerationConfig(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repeat_penalty=args.repeat_penalty,
                repeat_last_n=args.repeat_last_n,
                timeout_seconds=args.timeout,
            ),
        )
        if not response.strip():
            raise RuntimeError("Model returned an empty response.")
        print(f"[model-test] OK backend={args.backend} model={args.model}")
        print(response.strip())
        return 0
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


def _print_help() -> None:
    pipeline_cli.build_parser().print_help()
    print("\nAdditional command:")
    print("  model-test BACKEND MODEL   Test one model/backend without the pipeline")
    print("\nExample:")
    print("  mercan-sop model-test ollama qwen2.5:7b")


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "model-test":
        try:
            return cmd_model_test(raw[1:])
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if raw in (["-h"], ["--help"]):
        _print_help()
        return 0

    return pipeline_cli.main(raw)


if __name__ == "__main__":
    raise SystemExit(main())
