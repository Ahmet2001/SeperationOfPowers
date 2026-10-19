import re
import tempfile
import unittest
from pathlib import Path

from runtime.local_tools import (
    BUILTIN_TOOL_REGISTRY,
    READ_CHAR_BUDGET,
    execute_builtin_local,
)
from runtime.tool_validation import validate_tool_arguments
from yasama.yasama import CANONICAL_ACTIONS, YASAMA_SYSTEM_PROMPT


def _calculate(**arguments):
    return execute_builtin_local("CALCULATE", arguments)


class CalculateTests(unittest.TestCase):
    def test_four_operations_of_the_training_data(self) -> None:
        self.assertEqual(_calculate(operation="add", a=577, b=66111)["data"], {"result": 66688})
        self.assertEqual(_calculate(operation="subtract", a=56.89, b=9.2)["data"], {"result": 47.69})
        self.assertEqual(_calculate(operation="multiply", a=25, b=4)["data"], {"result": 100})
        self.assertEqual(
            _calculate(operation="divide", a=74.19, b=81.39)["data"], {"result": 0.911537043863}
        )

    def test_sqrt_and_power(self) -> None:
        self.assertEqual(_calculate(operation="sqrt", a=144)["data"], {"result": 12})
        self.assertEqual(_calculate(operation="sqrt", a=2)["data"], {"result": 1.41421356237})
        self.assertEqual(_calculate(operation="power", a=2, b=10)["data"], {"result": 1024})

    def test_huge_integers_use_scientific_notation(self) -> None:
        self.assertEqual(_calculate(operation="power", a=2, b=64)["data"], {"result": 18446744073709551616})
        self.assertEqual(_calculate(operation="power", a=300, b=87)["data"], {"result": 3.23257909929e+215})
        self.assertEqual(_calculate(operation="power", a=10, b=400)["status"], "error")

    def test_binary_noise_and_integral_floats_are_cleaned(self) -> None:
        self.assertEqual(_calculate(operation="add", a=0.1, b=0.2)["data"], {"result": 0.3})
        self.assertEqual(_calculate(operation="divide", a=10, b=2)["data"], {"result": 5})

    def test_invalid_requests_are_error_observations(self) -> None:
        cases = [
            {"operation": "divide", "a": 1, "b": 0},
            {"operation": "sqrt", "a": -4},
            {"operation": "power", "a": 9, "b": 10**6},
            {"operation": "power", "a": 0, "b": -1},
            {"operation": "modulo", "a": 7, "b": 2},
            {"operation": "add", "a": 1},
            {"operation": "add", "a": True, "b": 2},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                observation = _calculate(**arguments)
                self.assertEqual(observation["status"], "error")
                self.assertTrue(observation["error"])

    def test_schema_accepts_sqrt_without_b(self) -> None:
        schema = BUILTIN_TOOL_REGISTRY["CALCULATE"]
        self.assertEqual(
            validate_tool_arguments("CALCULATE", {"operation": "sqrt", "a": 9}, schema),
            {"operation": "sqrt", "a": 9},
        )

    def test_yasama_can_choose_calculate(self) -> None:
        self.assertIn("CALCULATE", CANONICAL_ACTIONS)
        self.assertIn("CALCULATE,", YASAMA_SYSTEM_PROMPT)
        self.assertIn("- CALCULATE: İki sayı", YASAMA_SYSTEM_PROMPT)


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

    def test_binary_and_missing_files_are_errors(self) -> None:
        (self.root / "logo.png").write_bytes(b"\x89PNG\0\0\0")
        for path in ("logo.png", "yok.txt"):
            with self.subTest(path=path):
                self.assertEqual(self._read(path)["status"], "error")


if __name__ == "__main__":
    unittest.main()
