"""Yargi receives the real runtime capabilities rather than hardcoded claims."""

from __future__ import annotations

import json
import unittest

import cli
from yargi.yargi import YargiRuntime


class FakeBackend:
    def __init__(self):
        self.last_messages = None

    def generate(self, messages, *, config=None):
        self.last_messages = messages
        return "Yanıt"


class RuntimeInfoTests(unittest.TestCase):
    def test_cli_exposes_configured_roles_and_executor_status_to_yargi(self):
        config = json.loads(json.dumps(cli.DEFAULT_CONFIG))
        for role in ("yasama", "yurutme", "yargi"):
            config["roles"][role]["model"] = "qwen3:1.7b"
        pipeline, _ = cli.build_runtime(config)
        info = pipeline.yargi._runtime_info
        self.assertEqual(info["roles"]["yasama"]["model"], "qwen3:1.7b")
        self.assertTrue(info["tools"]["FILE_SEARCH"]["built_in"])
        self.assertFalse(info["tools"]["SEND_MAIL"]["built_in"])
        self.assertTrue(info["tools"]["SEND_MAIL"]["external_executor_needed"])
        self.assertFalse(info["tools"]["SEND_MAIL"]["external_executor_configured"])
        self.assertNotIn("WEB_SEARCH", info["tools"])

    def test_yargi_system_context_reflects_actual_tools(self):
        backend = FakeBackend()
        yargi = YargiRuntime(backend=backend)
        yargi.set_runtime_info({
            "roles": {"yargi": {"model": "qwen3:1.7b", "backend": "ollama"}},
            "tools": {"FILE_READ": {"built_in": True}},
        })
        response = yargi.run(
            user_prompt="Hangi araçları kullanabiliyorsun?",
            conversation_history=[],
            observations=[],
        )
        self.assertEqual(response, "Yanıt")
        system = backend.last_messages[0]["content"]
        self.assertIn("RUNTIME_INFO", system)
        self.assertIn('"FILE_READ"', system)
        self.assertIn('"qwen3:1.7b"', system)
        self.assertNotIn("SEND_MAIL", system.split("RUNTIME_INFO:")[1])


if __name__ == "__main__":
    unittest.main()
