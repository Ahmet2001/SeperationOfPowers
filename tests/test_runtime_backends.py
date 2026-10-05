from __future__ import annotations

import unittest

from runtime import GenerationConfig, MercanCliBackend
from yasama.yasama import YasamaRuntime
from yargi.yargi import YargiRuntime
from yurutme.yurutme import YurutmeRuntime


class FakeBackend:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[list[dict[str, str]], GenerationConfig | None]] = []

    def generate(self, messages, *, config=None):
        normalized = [dict(message) for message in messages]
        self.calls.append((normalized, config))
        return self.response


class RuntimeBackendTests(unittest.TestCase):
    def test_yasama_uses_backend_and_returns_single_action(self) -> None:
        backend = FakeBackend("FILE_READ")
        runtime = YasamaRuntime(backend=backend)

        action = runtime.run(
            user_prompt="main.py dosyasini oku",
            conversation_history=[],
            state={"completed_actions": [], "last_observation": None},
        )

        self.assertEqual(action, "FILE_READ")
        self.assertEqual(backend.calls[0][0][0]["role"], "system")
        self.assertEqual(backend.calls[0][0][1]["role"], "user")

    def test_yurutme_parses_backend_json(self) -> None:
        backend = FakeBackend('{"path":"main.py"}')
        runtime = YurutmeRuntime(backend=backend)

        arguments = runtime.run(
            user_prompt="main.py dosyasini oku",
            conversation_history=[],
            state={"completed_actions": [], "last_observation": None},
            action="FILE_READ",
            tool_schema={"parameters": {"path": {"type": "string"}}},
            placeholder={"path": None},
        )

        self.assertEqual(arguments, {"path": "main.py"})

    def test_yargi_returns_natural_language(self) -> None:
        backend = FakeBackend("main.py dosyasi okundu.")
        runtime = YargiRuntime(backend=backend)

        response = runtime.run(
            user_prompt="main.py dosyasini oku",
            conversation_history=[],
            observations=[
                {
                    "action": "FILE_READ",
                    "data": {"path": "main.py"},
                    "status": "success",
                    "error": None,
                }
            ],
        )

        self.assertEqual(response, "main.py dosyasi okundu.")

    def test_mercan_cli_log_prefixes_are_removed(self) -> None:
        raw = (
            "Using cached model: /tmp/model.mercan\n"
            "Backend: CPU\n"
            "Loading /tmp/model.mercan...\n"
            "FILE_READ\n"
        )
        self.assertEqual(MercanCliBackend._strip_cli_logs(raw), "FILE_READ")


if __name__ == "__main__":
    unittest.main()
