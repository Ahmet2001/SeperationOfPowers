"""Regression coverage for schema-constrained Yurutme and FILE_SEARCH."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main import AgentPipeline
from runtime.local_tools import BUILTIN_TOOL_REGISTRY
from runtime.ollama_backend import OllamaBackend
from runtime.tool_validation import validate_tool_arguments
from yurutme.yurutme import YurutmeRuntime


class QueueBackend:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def generate(self, messages, *, config=None):
        self.calls.append([dict(item) for item in messages])
        return self.answers.pop(0)


class FakeRouter:
    def run(self, *, user_prompt, conversation_history, state):
        return "FINISH" if state["completed_actions"] else "FILE_SEARCH"


class FakeJudge:
    def run(self, *, user_prompt, conversation_history, observations):
        if not observations:
            return "Arama yapılmadı."
        return ", ".join(
            match["path"] for match in observations[0]["data"]["matches"]
        )


class StructuredToolTests(unittest.TestCase):
    def test_ollama_passes_tool_json_schema_to_format(self):
        backend = OllamaBackend("qwen3:1.7b")
        schema = BUILTIN_TOOL_REGISTRY["FILE_SEARCH"]["parameters"]
        fake_response = {"message": {"content": '{"query":"OllamaBackend"}'}}
        with patch(
            "runtime.ollama_backend._post_json", return_value=fake_response
        ) as post:
            text = backend.generate_structured(
                [{"role": "user", "content": "find OllamaBackend"}],
                schema=schema,
            )
        self.assertIn("OllamaBackend", text)
        payload = post.call_args.args[1]
        self.assertEqual(payload["format"], schema)
        self.assertIs(payload["think"], False)

    def test_yurutme_recovers_from_echoed_tool_schema(self):
        invalid = '{"action":"FILE_SEARCH","tool_schema":{},"placeholder":{}}'
        valid = '{"query":"OllamaBackend","path":".","pattern":"*.py","recursive":true,"max_results":50}'
        backend = QueueBackend([invalid, valid])
        runtime = YurutmeRuntime(backend=backend)
        args = runtime.run(
            user_prompt='Projede "OllamaBackend" geçen Python dosyalarını ara.',
            conversation_history=[],
            state={"completed_actions": []},
            action="FILE_SEARCH",
            tool_schema=BUILTIN_TOOL_REGISTRY["FILE_SEARCH"],
            placeholder={"query": None, "path": ".", "pattern": "*.py"},
        )
        self.assertEqual(args["query"], "OllamaBackend")
        self.assertEqual(args["pattern"], "*.py")
        self.assertEqual(len(backend.calls), 2)
        self.assertIn("Önceki JSON yanlış", backend.calls[1][-1]["content"])

    def test_yurutme_stops_after_bounded_invalid_retry(self):
        echoed = '{"action":"FILE_SEARCH","tool_schema":{}}'
        runtime = YurutmeRuntime(backend=QueueBackend([echoed, echoed]))
        with self.assertRaisesRegex(ValueError, "could not produce valid FILE_SEARCH"):
            runtime.run(
                user_prompt='Projede "OllamaBackend" ara.',
                conversation_history=[],
                state={},
                action="FILE_SEARCH",
                tool_schema=BUILTIN_TOOL_REGISTRY["FILE_SEARCH"],
            )

    def test_validator_rejects_missing_and_extra_fields(self):
        schema = BUILTIN_TOOL_REGISTRY["FILE_SEARCH"]
        with self.assertRaisesRegex(ValueError, "unexpected argument fields"):
            validate_tool_arguments(
                "FILE_SEARCH",
                {"action": "FILE_SEARCH", "tool_schema": {}},
                schema,
            )
        with self.assertRaisesRegex(ValueError, "missing required"):
            validate_tool_arguments("FILE_SEARCH", {"path": "."}, schema)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            validate_tool_arguments("FILE_SEARCH", {"query": ""}, schema)

    def test_pipeline_rejects_bad_args_before_executor(self):
        class InvalidRunner:
            def run(self, **kwargs):
                return {"action": "FILE_SEARCH", "tool_schema": {}}

        calls = []
        pipeline = AgentPipeline(
            yasama=FakeRouter(),
            yurutme=InvalidRunner(),
            yargi=FakeJudge(),
            tool_registry={},
            executor=lambda action, args: calls.append((action, args)),
        )
        with self.assertRaisesRegex(ValueError, "unexpected argument fields"):
            pipeline.run('Projede "OllamaBackend" ara.')
        self.assertEqual(calls, [])

    def test_end_to_end_keyword_retrieval_still_works(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "my_backend.py").write_text(
                "class OllamaBackend:\n    pass\n", encoding="utf-8"
            )

            class ValidRunner:
                def run(self, **kwargs):
                    return {
                        "query": "OllamaBackend",
                        "path": ".",
                        "pattern": "*.py",
                        "recursive": True,
                        "max_results": 50,
                    }

            pipeline = AgentPipeline(
                yasama=FakeRouter(),
                yurutme=ValidRunner(),
                yargi=FakeJudge(),
                tool_registry={},
                executor=lambda *_: self.fail("No external executor needed"),
            )
            from runtime import local_tools
            original = local_tools.execute_builtin_local
            with patch(
                "main.execute_builtin_local",
                side_effect=lambda action, args: original(
                    action, args, workspace=root
                ),
            ):
                result = pipeline.run('Projede "OllamaBackend" geçen dosyayı bul.')
            self.assertIn("my_backend.py", result)


if __name__ == "__main__":
    unittest.main()
