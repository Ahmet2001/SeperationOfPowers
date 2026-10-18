from __future__ import annotations

import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch
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

    def test_libmercan_delegates_messages_options_and_lifetime_to_sdk(self) -> None:
        sdk = ModuleType("ethosoftlib.mercan")
        sdk.MercanRuntime = MagicMock()
        sdk.Model = MagicMock()
        sdk.Model.return_value.generate.return_value = "FILE_READ"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.mercan"
            path.touch()
            with patch.dict("sys.modules", {"ethosoftlib.mercan": sdk}):
                backend = LibMercanBackend(path, library="/test/libmercan.so", threads=2,
                                           threads_batch=3, plugins=["/test/plugin.so"])
                runtime = sdk.MercanRuntime.return_value
                runtime.load_plugin.assert_called_once_with("/test/plugin.so")
                sdk.MercanRuntime.assert_called_once_with(library_path="/test/libmercan.so")
                self.assertEqual(sdk.Model.call_args.kwargs["threads_batch"], 3)
                self.assertEqual(sdk.Model.call_args.kwargs["system_prompt"], "")
                messages = [{"role": "system", "content": "Yasama"},
                            {"role": "user", "content": "Dosyayı oku"}]
                self.assertEqual(backend.generate(messages, config=GenerationConfig(max_tokens=12)), "FILE_READ")
                sdk.Model.return_value.generate.assert_called_once_with(
                    messages, max_tokens=12, temperature=0.0, top_k=40, top_p=1.0,
                    repeat_penalty=1.0, repeat_last_n=64)
                backend.close()
                backend.close()
                sdk.Model.return_value.close.assert_called_once()
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    backend.generate(messages)

    def test_libmercan_releases_runtime_if_model_load_fails(self) -> None:
        sdk = ModuleType("ethosoftlib.mercan")
        sdk.MercanRuntime = MagicMock()
        sdk.Model = MagicMock()
        sdk.MercanRuntime.return_value.load_model.side_effect = RuntimeError("invalid model")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.mercan"
            path.touch()
            with patch.dict("sys.modules", {"ethosoftlib.mercan": sdk}):
                with self.assertRaisesRegex(RuntimeError, "invalid model"):
                    LibMercanBackend(path)
            sdk.MercanRuntime.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
