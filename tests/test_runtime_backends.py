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
from yasama.yasama import CANONICAL_ACTIONS, YasamaRuntime
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
        messages = backend.calls[0][0]
        self.assertEqual(messages[0]["role"], "system")
        # Few-shot examples precede the request as alternating user/assistant
        # turns whose answers are bare canonical actions.
        examples = messages[1:-1]
        self.assertTrue(examples)
        self.assertEqual(
            [m["role"] for m in examples],
            ["user", "assistant"] * (len(examples) // 2),
        )
        for answer in examples[1::2]:
            self.assertIn(answer["content"], CANONICAL_ACTIONS)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("USER:\nmain.py dosyasını oku", messages[-1]["content"])
        self.assertIn("STATE:\n{", messages[-1]["content"])

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
        messages = backend.calls[0][0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(
            [m["role"] for m in messages[1:-1]], ["user", "assistant"] * 2
        )
        self.assertIn("ACTION:\nCODE_EXECUTE", messages[1]["content"])
        self.assertIn("ACTION:\nFILE_READ", messages[3]["content"])
        prompt = messages[-1]["content"]
        self.assertIn("USER:\nmain.py dosyasını oku", prompt)
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
        messages = backend.calls[0][0]
        examples = messages[1:-1]
        self.assertEqual(
            [m["role"] for m in examples],
            ["user", "assistant"] * (len(examples) // 2),
        )
        # Examples must cover both outcomes so the model does not learn to
        # report a single one regardless of the real observation.
        self.assertIn('"status": "success"', "".join(m["content"] for m in examples))
        self.assertIn('"status": "error"', "".join(m["content"] for m in examples))
        prompt = messages[-1]["content"]
        self.assertTrue(prompt.startswith("main.py dosyasını oku\n<tool_observations>\n["))
        self.assertTrue(prompt.endswith("\n</tool_observations>"))

    def test_yargi_replays_history_as_chat_turns_without_examples(self) -> None:
        backend = FakeBackend("Ankara.")
        runtime = YargiRuntime(backend=backend)

        runtime.run(
            user_prompt="Türkiye'nin başkenti neresi?",
            conversation_history=[
                {"role": "user", "content": "selam"},
                {"role": "assistant", "content": "Selam!"},
            ],
            observations=[],
        )

        messages = backend.calls[0][0]
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user"])
        self.assertTrue(messages[1]["content"].startswith("selam\n<tool_observations>"))
        self.assertEqual(messages[2]["content"], "Selam!")
        # The history must not also be pasted into the request as JSON.
        self.assertNotIn("CONVERSATION_HISTORY", messages[-1]["content"])
        self.assertTrue(messages[-1]["content"].startswith("Türkiye'nin başkenti neresi?\n"))

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
        sdk.Model.NATIVE_GENERATION_API_VERSION = 1
        sdk.Model.DEVICE_API_VERSION = 1
        sdk.Model.return_value.generate.return_value = "FILE_READ"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.mercan"
            path.touch()
            with patch.dict("sys.modules", {"ethosoftlib.mercan": sdk}):
                backend = LibMercanBackend(path, library="/test/libmercan.so", device="cuda", threads=2,
                                           threads_batch=3, plugins=["/test/plugin.so"])
                runtime = sdk.MercanRuntime.return_value
                runtime.load_plugin.assert_called_once_with("/test/plugin.so")
                self.assertEqual(runtime.load_model.call_args.kwargs["device"], "cuda")
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
        sdk.Model.NATIVE_GENERATION_API_VERSION = 1
        sdk.Model.DEVICE_API_VERSION = 1
        sdk.MercanRuntime.return_value.load_model.side_effect = RuntimeError("invalid model")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.mercan"
            path.touch()
            with patch.dict("sys.modules", {"ethosoftlib.mercan": sdk}):
                with self.assertRaisesRegex(RuntimeError, "invalid model"):
                    LibMercanBackend(path)
            sdk.MercanRuntime.return_value.close.assert_called_once()

    def test_libmercan_cuts_plain_text_chatml_end_marker(self) -> None:
        sdk = ModuleType("ethosoftlib.mercan")
        sdk.MercanRuntime = MagicMock()
        sdk.Model = MagicMock()
        sdk.Model.NATIVE_GENERATION_API_VERSION = 1
        sdk.Model.DEVICE_API_VERSION = 1
        sdk.Model.return_value.generate.return_value = "RESPOND<|im_end|>\n<|im_start|>kullanici"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.mercan"
            path.touch()
            with patch.dict("sys.modules", {"ethosoftlib.mercan": sdk}):
                with LibMercanBackend(path) as backend:
                    self.assertEqual(
                        backend.generate([{"role": "user", "content": "Selam"}]),
                        "RESPOND",
                    )


if __name__ == "__main__":
    unittest.main()
