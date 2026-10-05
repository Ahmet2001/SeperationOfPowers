from __future__ import annotations

import ctypes
import random
import unittest

from runtime import GenerationConfig, LibMercanBackend, MercanCliBackend
from runtime.serialization import (
    serialize_yargi_input,
    serialize_yasama_input,
    serialize_yurutme_input,
)
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
            user_prompt="main.py dosyasını oku",
            conversation_history=[],
            state={"completed_actions": [], "last_observation": None},
        )

        self.assertEqual(action, "FILE_READ")
        self.assertEqual(backend.calls[0][0][0]["role"], "system")
        self.assertEqual(backend.calls[0][0][1]["role"], "user")
        self.assertIn("USER:\nmain.py dosyasını oku", backend.calls[0][0][1]["content"])
        self.assertIn("STATE:\n{", backend.calls[0][0][1]["content"])

    def test_yurutme_parses_backend_json(self) -> None:
        backend = FakeBackend('{"path":"main.py"}')
        runtime = YurutmeRuntime(backend=backend)

        arguments = runtime.run(
            user_prompt="main.py dosyasını oku",
            conversation_history=[],
            state={"completed_actions": [], "last_observation": None},
            action="FILE_READ",
            tool_schema={"parameters": {"path": {"type": "string"}}},
            placeholder={"path": None},
        )

        self.assertEqual(arguments, {"path": "main.py"})
        prompt = backend.calls[0][0][1]["content"]
        self.assertIn("ACTION:\nFILE_READ", prompt)
        self.assertIn("TOOL_SCHEMA:\n{", prompt)
        self.assertIn('PLACEHOLDER:\n{\n  "path": null\n}', prompt)

    def test_yargi_returns_natural_language(self) -> None:
        backend = FakeBackend("main.py dosyası okundu.")
        runtime = YargiRuntime(backend=backend)

        response = runtime.run(
            user_prompt="main.py dosyasını oku",
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

        self.assertEqual(response, "main.py dosyası okundu.")
        prompt = backend.calls[0][0][1]["content"]
        self.assertTrue(prompt.startswith("main.py dosyasını oku\n<tool_observations>\n["))
        self.assertTrue(prompt.endswith("\n</tool_observations>"))

    def test_yasama_serializer_matches_documented_layout(self) -> None:
        prompt = serialize_yasama_input(
            user_prompt="notes.txt dosyasını oluştur ve içine Merhaba yaz.",
            conversation_history=[],
            state={"completed_actions": [], "last_observation": None},
        )
        self.assertEqual(
            prompt,
            "USER:\nnotes.txt dosyasını oluştur ve içine Merhaba yaz.\n"
            "STATE:\n{\n"
            '  "completed_actions": [],\n'
            '  "last_observation": null\n'
            "}",
        )

    def test_yurutme_serializer_keeps_target_out_of_prompt(self) -> None:
        prompt = serialize_yurutme_input(
            user_prompt="notes.txt dosyasını oluştur ve içine Merhaba yaz.",
            conversation_history=[],
            state={"completed_actions": [], "last_observation": None},
            action="FILE_CREATE",
            tool_schema={
                "parameters": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                }
            },
            placeholder={"path": None, "content": None},
        )
        self.assertIn("USER:\nnotes.txt dosyasını oluştur", prompt)
        self.assertIn("ACTION:\nFILE_CREATE", prompt)
        self.assertIn("PLACEHOLDER:\n{", prompt)
        self.assertNotIn('"content": "Merhaba"', prompt)

    def test_yargi_serializer_uses_tool_observations_block(self) -> None:
        prompt = serialize_yargi_input(
            user_prompt="notes.txt dosyasını oluştur ve içine Merhaba yaz.",
            conversation_history=[],
            observations=[
                {
                    "action": "FILE_CREATE",
                    "data": {"path": "notes.txt", "content": "Merhaba"},
                    "status": "success",
                    "error": None,
                }
            ],
        )
        self.assertEqual(
            prompt.splitlines()[0],
            "notes.txt dosyasını oluştur ve içine Merhaba yaz.",
        )
        self.assertIn("<tool_observations>\n[", prompt)
        self.assertIn('"action": "FILE_CREATE"', prompt)
        self.assertTrue(prompt.endswith("</tool_observations>"))

    def test_mercan_cli_log_prefixes_are_removed(self) -> None:
        raw = (
            "Using cached model: /tmp/model.mercan\n"
            "Backend: CPU\n"
            "Loading /tmp/model.mercan...\n"
            "FILE_READ\n"
        )
        self.assertEqual(MercanCliBackend._strip_cli_logs(raw), "FILE_READ")

    def test_libmercan_chat_format_matches_mercan_cli_roles(self) -> None:
        formatted = LibMercanBackend._format_chat(
            [
                {"role": "system", "content": "Sistem"},
                {"role": "user", "content": "Merhaba"},
            ]
        )
        self.assertEqual(
            formatted,
            "<|im_start|>sistem\nSistem<|im_end|>\n"
            "<|im_start|>kullanici\nMerhaba<|im_end|>\n"
            "<|im_start|>asistan\n",
        )

    def test_libmercan_greedy_sampling_chooses_highest_logit(self) -> None:
        logits = (ctypes.c_float * 3)(0.1, 3.0, 1.5)
        token = LibMercanBackend._sample_token(
            logits_ptr=logits,
            n_vocab=3,
            temperature=0.0,
            top_k=3,
            top_p=1.0,
            repeat_penalty=1.0,
            recent_tokens=[],
            rng=random.Random(0),
        )
        self.assertEqual(token, 1)


if __name__ == "__main__":
    unittest.main()
