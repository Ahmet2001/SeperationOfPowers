from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from main import AgentPipeline
from runtime.output_parsing import (
    parse_canonical_action,
    parse_json_object,
    unwrap_natural_language_response,
)
from yasama.yasama import CANONICAL_ACTIONS


class GenericCompatibilityTests(unittest.TestCase):
    def test_yasama_accepts_labeled_action(self) -> None:
        self.assertEqual(
            parse_canonical_action(
                "ACTION: WEB_OPEN\n\nEXPLANATION: generic model wrapper",
                CANONICAL_ACTIONS,
            ),
            "WEB_OPEN",
        )

    def test_yurutme_accepts_fenced_json(self) -> None:
        self.assertEqual(
            parse_json_object('```json\n{"path":"main.py"}\n```'),
            {"path": "main.py"},
        )

    def test_yargi_unwraps_response_object(self) -> None:
        self.assertEqual(
            unwrap_natural_language_response('{"response":"Merhaba"}'),
            "Merhaba",
        )

    def test_pipeline_uses_builtin_file_list_when_registry_is_empty(self) -> None:
        class FakeYasama:
            def __init__(self) -> None:
                self.calls = 0

            def run(self, **kwargs):
                self.calls += 1
                return "FILE_LIST" if self.calls == 1 else "FINISH"

        class FakeYurutme:
            def run(self, **kwargs):
                return {"path": ".", "pattern": "*.py", "recursive": False}

        class FakeYargi:
            def run(self, **kwargs):
                observations = kwargs["observations"]
                self_observation = observations[0]
                return ",".join(self_observation["data"]["entries"])

        def should_not_run(action, arguments):
            raise AssertionError("external executor should not run for built-in FILE_LIST")

        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = Path.cwd()
            try:
                Path(temp_dir, "a.py").write_text("print('a')", encoding="utf-8")
                Path(temp_dir, "b.txt").write_text("b", encoding="utf-8")
                import os

                os.chdir(temp_dir)
                pipeline = AgentPipeline(
                    yasama=FakeYasama(),
                    yurutme=FakeYurutme(),
                    yargi=FakeYargi(),
                    tool_registry={},
                    executor=should_not_run,
                    max_steps=3,
                )
                self.assertEqual(pipeline.run("python dosyalarını listele"), "a.py")
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
