from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from runtime.local_tools import BUILTIN_TOOL_REGISTRY, execute_builtin_local
from runtime.output_parsing import unwrap_natural_language_response


class DefaultToolTests(unittest.TestCase):
    def test_file_list_glob_filters_python_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("print('a')\n", encoding="utf-8")
            (root / "b.txt").write_text("b\n", encoding="utf-8")

            observation = execute_builtin_local(
                "FILE_LIST",
                {"path": ".", "pattern": "*.py", "recursive": False},
                workspace=root,
            )

        self.assertEqual(observation["status"], "success")
        self.assertEqual(observation["data"]["pattern"], "*.py")
        self.assertEqual(observation["data"]["entries"], ["a.py"])

    def test_file_list_placeholder_does_not_bias_pattern(self) -> None:
        self.assertIsNone(BUILTIN_TOOL_REGISTRY["FILE_LIST"]["placeholder"]["pattern"])
        description = BUILTIN_TOOL_REGISTRY["FILE_LIST"]["description"]
        self.assertIn("*.py", description)

    def test_send_mail_schema_exists_but_has_no_builtin_executor(self) -> None:
        schema = BUILTIN_TOOL_REGISTRY["SEND_MAIL"]
        self.assertTrue(schema["_executor_required"])
        self.assertNotIn("_builtin_executor", schema)
        self.assertEqual(
            schema["parameters"]["required"],
            ["to", "subject", "body"],
        )

    def test_yargi_simple_result_list_is_rendered_as_text(self) -> None:
        raw = '{"result":["cli.py","main.py"]}'
        self.assertEqual(
            unwrap_natural_language_response(raw),
            "- cli.py\n- main.py",
        )


if __name__ == "__main__":
    unittest.main()
