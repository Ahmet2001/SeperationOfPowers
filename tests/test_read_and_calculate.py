import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime.local_tools import (
    BUILTIN_TOOL_REGISTRY,
    READ_CHAR_BUDGET,
    execute_builtin_local,
)
from runtime.tool_validation import validate_tool_arguments
from yasama.yasama import CANONICAL_ACTIONS, YASAMA_SYSTEM_PROMPT


def _calculate(expression):
    return execute_builtin_local("CALCULATE", {"expression": expression})


class CalculateTests(unittest.TestCase):
    def test_expressions_of_the_training_data(self) -> None:
        # tool-math `calc` expressions and converted calculator operations.
        cases = {
            "15*0.6": 9,
            "1462.5/37.5": 39,
            "577+66111": 66688,
            "56.89-9.2": 47.69,
            "74.19/81.39": 0.911537043863,
            "3*12.5+2*4": 45.5,
        }
        for expression, result in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(_calculate(expression)["data"], {"result": result})

    def test_roots_powers_and_percentages(self) -> None:
        self.assertEqual(_calculate("sqrt(625)")["data"], {"result": 25})
        self.assertEqual(_calculate("sqrt(2)")["data"], {"result": 1.41421356237})
        self.assertEqual(_calculate("2^16")["data"], {"result": 65536})
        self.assertEqual(_calculate("2**10")["data"], {"result": 1024})
        self.assertEqual(_calculate("50'nin %20'si")["data"], {"result": 10})

    def test_huge_integers_use_scientific_notation(self) -> None:
        self.assertEqual(_calculate("2^64")["data"], {"result": 18446744073709551616})
        self.assertEqual(_calculate("300^87")["data"], {"result": 3.23257909929e+215})
        self.assertEqual(_calculate("10^400")["status"], "error")

    def test_binary_noise_and_integral_results_are_cleaned(self) -> None:
        self.assertEqual(_calculate("0.1+0.2")["data"], {"result": 0.3})
        self.assertEqual(_calculate("10/2")["data"], {"result": 5})
        self.assertEqual(_calculate("340/80")["data"], {"result": 4.25})

    def test_invalid_expressions_are_error_observations(self) -> None:
        for expression in ("1/0", "x+1", "2^100000", "", "__import__('os')", "import os"):
            with self.subTest(expression=expression):
                observation = _calculate(expression)
                self.assertEqual(observation["status"], "error")
                self.assertTrue(observation["error"])

    def test_schema_takes_one_expression(self) -> None:
        schema = BUILTIN_TOOL_REGISTRY["CALCULATE"]
        self.assertEqual(
            validate_tool_arguments("CALCULATE", {"expression": "2+2"}, schema),
            {"expression": "2+2"},
        )
        with self.assertRaises(ValueError):
            validate_tool_arguments("CALCULATE", {"operation": "add", "a": 1, "b": 2}, schema)

    def test_yasama_can_choose_calculate(self) -> None:
        self.assertIn("CALCULATE", CANONICAL_ACTIONS)
        self.assertIn("CALCULATE,", YASAMA_SYSTEM_PROMPT)
        self.assertIn("- CALCULATE: Bir matematik ifadesini", YASAMA_SYSTEM_PROMPT)


class FileReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _read(self, path, **kwargs):
        user_prompt = kwargs.pop("user_prompt", None)
        return execute_builtin_local(
            "FILE_READ", {"path": path, **kwargs}, workspace=self.root, user_prompt=user_prompt
        )

    def _long_file(self) -> str:
        filler = "\n".join(f"Bu satır {i} numaralı dolgu metnidir ve konu dışıdır." for i in range(600))
        target = "Docker kurulumu için önce docker paketini yükleyin ve servisi başlatın."
        text = filler + "\n" + target + "\n" + filler
        (self.root / "kurulum.md").write_text(text, encoding="utf-8")
        return target

    def test_short_file_is_returned_whole(self) -> None:
        (self.root / "VERSION").write_text("2.4.1\n", encoding="utf-8")
        self.assertEqual(self._read("VERSION")["data"], {"path": "VERSION", "content": "2.4.1\n"})

    def test_long_file_returns_matching_sections_within_budget(self) -> None:
        target = self._long_file()
        data = self._read("kurulum.md", query="Docker kurulumu")["data"]
        self.assertTrue(data["matched"])
        self.assertTrue(data["partial"])
        self.assertGreater(data["total_chars"], READ_CHAR_BUDGET)
        self.assertIn(target, data["content"])
        lines = [int(n) for n in re.findall(r"^\[satır (\d+)\]$", data["content"], re.M)]
        self.assertTrue(lines)
        self.assertEqual(lines, sorted(lines))
        body = re.sub(r"^\[satır \d+\]\n", "", data["content"], flags=re.M)
        self.assertLessEqual(len(body), READ_CHAR_BUDGET + 2 * len(lines))

    def test_user_prompt_is_the_query_when_yurutme_gives_none(self) -> None:
        target = self._long_file()
        data = self._read("kurulum.md", user_prompt="kurulum.md dosyasında Docker kurulumu nasıl yapılır?")["data"]
        # The written file name and request words are not search terms.
        self.assertEqual(data["query"], "docke kurul yapıl")
        self.assertIn(target, data["content"])

    def test_no_match_falls_back_to_the_start_of_the_file(self) -> None:
        self._long_file()
        data = self._read("kurulum.md", query="Kubernetes")["data"]
        self.assertFalse(data["matched"])
        self.assertTrue(data["content"].startswith("[satır 1]\n"))

    def test_short_file_needs_no_ethosoftlib(self) -> None:
        (self.root / "VERSION").write_text("2.4.1\n", encoding="utf-8")
        blocked = {name: None for name in ("ethosoftlib", "ethosoftlib.rag")}
        with patch.dict(sys.modules, blocked):
            self.assertEqual(self._read("VERSION")["status"], "success")

    def test_binary_and_missing_files_are_errors(self) -> None:
        (self.root / "logo.png").write_bytes(b"\x89PNG\0\0\0")
        for path in ("logo.png", "yok.txt"):
            with self.subTest(path=path):
                self.assertEqual(self._read(path)["status"], "error")


if __name__ == "__main__":
    unittest.main()
