"""Backend-neutral text generation adapters.

Role runtimes depend only on ``ChatBackend``.  The concrete model transport can
therefore be changed without touching Yasama/Yurutme/Yargi orchestration.

Supported backends:
- Ollama native HTTP API
- llama.cpp OpenAI-compatible server API
- Mercan CLI, which is linked against the custom libmercan runtime
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ChatMessage = Mapping[str, str]


@dataclass(frozen=True)
class GenerationConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 40
    repeat_penalty: float = 1.0
    repeat_last_n: int = 64
    timeout_seconds: float = 120.0


class ChatBackend(Protocol):
    """Minimal interface required by all three role runtimes."""

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        ...


def _normalized_messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported chat role: {role!r}")
        if not isinstance(content, str):
            raise TypeError("Chat message content must be a string.")
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise ValueError("At least one chat message is required.")
    return normalized


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    timeout: float,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Model server returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach model server at {url}: {exc}") from exc

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Model server returned invalid JSON.") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("Model server response must be a JSON object.")
    return decoded


class OllamaBackend:
    """Ollama ``/api/chat`` backend."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        cfg = config or GenerationConfig()
        payload = {
            "model": self.model,
            "messages": _normalized_messages(messages),
            "stream": False,
            "options": {
                "num_predict": cfg.max_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "repeat_penalty": cfg.repeat_penalty,
                "repeat_last_n": cfg.repeat_last_n,
            },
        }
        response = _post_json(
            f"{self.base_url}/api/chat",
            payload,
            timeout=cfg.timeout_seconds,
        )
        message = response.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise RuntimeError("Ollama response is missing message.content.")
        return str(message["content"])


class LlamaCppBackend:
    """llama.cpp server using its OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080",
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        cfg = config or GenerationConfig()
        payload: dict[str, Any] = {
            "messages": _normalized_messages(messages),
            "stream": False,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        }
        if self.model:
            payload["model"] = self.model

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = _post_json(
            f"{self.base_url}/v1/chat/completions",
            payload,
            timeout=cfg.timeout_seconds,
            headers=headers,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("llama.cpp response is missing choices.")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise RuntimeError("llama.cpp response contains an invalid choice.")
        message = first.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise RuntimeError("llama.cpp response is missing choices[0].message.content.")
        return str(message["content"])


class MercanCliBackend:
    """Adapter for the custom Mercan runtime CLI backed by libmercan.

    ``mercanApp-test1`` builds ``mercan`` as an executable linked against
    ``libmercan``.  Using that executable keeps this Python project independent
    of whether libmercan itself was built static or shared.
    """

    _LOG_PREFIXES = (
        "Backend:",
        "Loading ",
        "Using cached model:",
        "Pulling ",
        "Saved:",
    )

    def __init__(
        self,
        model: str,
        *,
        binary: str = "mercan",
        threads: int | None = None,
        gpu_layers: int | None = None,
        plugins: Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self.binary = binary
        self.threads = threads
        self.gpu_layers = gpu_layers
        self.plugins = list(plugins or [])

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        cfg = config or GenerationConfig()
        system_prompt, user_prompt = self._to_cli_prompt(messages)

        command = [
            self.binary,
            "run",
            self.model,
            "-s",
            system_prompt,
            "-p",
            user_prompt,
            "-n",
            str(cfg.max_tokens),
            "--temperature",
            str(cfg.temperature),
            "--top-k",
            str(cfg.top_k),
            "--top-p",
            str(cfg.top_p),
            "--repeat-penalty",
            str(cfg.repeat_penalty),
            "--repeat-last-n",
            str(cfg.repeat_last_n),
        ]
        if self.threads is not None:
            command.extend(["--threads", str(self.threads)])
        if self.gpu_layers is not None:
            command.extend(["--gpu-layers", str(self.gpu_layers)])
        for plugin in self.plugins:
            command.extend(["--plugin", plugin])

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=cfg.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Mercan runtime executable {self.binary!r} was not found."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Mercan generation timed out.") from exc

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"Mercan runtime exited with code {completed.returncode}: {detail}"
            )

        output = self._strip_cli_logs(completed.stdout)
        if not output:
            raise RuntimeError("Mercan runtime returned an empty response.")
        return output

    @staticmethod
    def _to_cli_prompt(messages: Sequence[ChatMessage]) -> tuple[str, str]:
        normalized = _normalized_messages(messages)
        system_parts: list[str] = []
        conversation: list[dict[str, str]] = []
        for message in normalized:
            if message["role"] == "system":
                system_parts.append(message["content"])
            else:
                conversation.append(message)

        if not conversation:
            raise ValueError("Mercan CLI backend requires a non-system prompt.")

        if len(conversation) == 1 and conversation[0]["role"] == "user":
            prompt = conversation[0]["content"]
        else:
            prompt = "\n".join(
                f"{message['role']}: {message['content']}" for message in conversation
            )

        return "\n\n".join(system_parts), prompt

    @classmethod
    def _strip_cli_logs(cls, stdout: str) -> str:
        lines = stdout.splitlines()
        while lines and (
            not lines[0].strip()
            or any(lines[0].startswith(prefix) for prefix in cls._LOG_PREFIXES)
        ):
            lines.pop(0)
        return "\n".join(lines).strip()
