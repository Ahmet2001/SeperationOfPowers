import json
import unittest

from datatools.normalize import (
    MAX_CREATE_CONTENT,
    WRITE_AS_CREATE,
    Reject,
    calculator_expression,
    check_calculation,
    check_numbers_against_results,
    clean_response,
    load_aliases,
    new_file_write_steps,
    normalize_arguments,
    normalize_observation,
    normalize_state,
    parse_web_result_text,
)

ALIASES = load_aliases({"mapping": [
    {"source_action": "add_and_execute_jupyter_code_cell", "ontology_action": "CODE_EXECUTE",
     "mapping_status": "mapped_safe_alias"},
    {"source_action": "send_message", "ontology_action": None, "mapping_status": "unmapped_preserved"},
]})


class ArgumentTests(unittest.TestCase):
    def test_source_names_become_registry_names(self) -> None:
        cases = [
            ("FILE_LIST", {"dir_path": "src"}, {"path": "src"}),
            ("FILE_READ", {"file_path": "a.py", "limit": 50, "offset": 10}, {"path": "a.py"}),
            ("FILE_EDIT", {"file_path": "a.py", "old_text": "x", "new_text": "y"},
             {"path": "a.py", "old_text": "x", "new_text": "y"}),
            ("FILE_SEARCH", {"path": ".", "pattern": "TODO"}, {"path": ".", "query": "TODO"}),
            ("SHELL_EXECUTE", {"command": "ls", "timeout": 30}, {"command": "ls"}),
            ("SEND_MAIL", {"recipient": "a@b.c", "subject": "S", "message": "M"},
             {"to": "a@b.c", "subject": "S", "body": "M"}),
            ("DOWNLOAD", {"url": "https://x/y", "output_path": "y"}, {"url": "https://x/y", "path": "y"}),
            ("FILE_MOVE", {"source_path": "a", "destination_path": "b"}, {"source": "a", "destination": "b"}),
            ("WEB_SEARCH", {"q": "hava", "num": 10}, {"query": "hava", "max_results": 3}),
            ("CODE_EXECUTE", {"answer": None, "code": "print(1)"}, {"code": "print(1)", "language": "python"}),
            ("CALCULATE", {"operation": "subtract", "a": 5, "b": -3}, {"expression": "5-(-3)"}),
            ("CALCULATE", {"expression": "15*0.6"}, {"expression": "15*0.6"}),
        ]
        for action, source, expected in cases:
            with self.subTest(action=action, source=source):
                self.assertEqual(normalize_arguments(action, source), expected)

    def test_unfixable_arguments_are_rejected(self) -> None:
        for action, source, reason in [
            ("CODE_EXECUTE", {"answer": "42", "code": "x"}, "code_execute_answer_in_arguments"),
            ("DOWNLOAD", {"data": 1, "size": 2, "type": "x"}, "schema:"),
            ("CALCULATE", {"tokens": [1, 2]}, "schema:"),
            ("FINISH", {}, "not_a_tool_action"),
        ]:
            with self.subTest(action=action), self.assertRaises(Reject) as caught:
                normalize_arguments(action, source)
            self.assertTrue(caught.exception.reason.startswith(reason))

    def test_calculator_operations_convert_without_loss(self) -> None:
        self.assertEqual(calculator_expression({"operation": "divide", "a": 74.19, "b": 81.39}), "74.19/81.39")
        self.assertEqual(calculator_expression({"operation": "add", "a": 577, "b": 66111}), "577+66111")


class ObservationTests(unittest.TestCase):
    def test_web_search_text_becomes_results(self) -> None:
        text = "[1] https://a.example/x\nBirinci özet.\n\n[2] http://b.example\nİkinci\nsatır.\n\n[3] https://c\nÜç\n\n[4] https://d\nDört"
        self.assertEqual(parse_web_result_text(text), [
            {"url": "https://a.example/x", "snippet": "Birinci özet."},
            {"url": "http://b.example", "snippet": "İkinci\nsatır."},
            {"url": "https://c", "snippet": "Üç"},
        ])
        observation = normalize_observation(
            {"action": "web_search", "tool_call_id": "c1", "data": {"result": text}, "status": "success", "error": None},
            ALIASES, query="azelaik asit",
        )
        self.assertEqual(observation["action"], "WEB_SEARCH")
        self.assertEqual(observation["data"]["query"], "azelaik asit")
        self.assertEqual(len(observation["data"]["results"]), 3)
        self.assertNotIn("tool_call_id", observation)

    def test_hunter_wrappers_and_gemini_names(self) -> None:
        listed = normalize_observation({"action": "list_directory", "data": {
            "success": True, "result": ["session_state.json (964 bytes)"], "tool": "list_directory", "source": "builtin",
        }, "status": "success", "error": None}, ALIASES)
        self.assertEqual(listed["data"], {"entries": ["session_state.json"]})
        failed = normalize_observation({"action": "read_file", "data": {
            "success": False, "result": "No such file", "tool": "read_file", "source": "builtin",
        }, "status": "success", "error": None}, ALIASES)
        self.assertEqual((failed["status"], failed["data"], failed["error"]), ("error", {}, "No such file"))
        gemini = normalize_observation({"action": "FILE_LIST", "data": {"files": ["a.py"], "path": "."},
                                        "status": "success", "error": None}, ALIASES)
        self.assertEqual(gemini["data"], {"path": ".", "entries": ["a.py"]})

    def test_leaked_answers_and_unknown_tools_are_rejected(self) -> None:
        for action, reason in [("RESPOND", "observation_control_action"), ("write_file", "observation_unmapped_action")]:
            with self.subTest(action=action), self.assertRaises(Reject) as caught:
                normalize_observation({"action": action, "data": {}, "status": "success", "error": None}, ALIASES)
            self.assertEqual(caught.exception.reason, reason)

    def test_state_uses_action_names_and_full_observations(self) -> None:
        obs = {"action": "calc", "tool_call_id": "c1", "data": {"result": 12}, "status": "success", "error": None}
        state = normalize_state({"completed_actions": [{"action": "calc"}], "last_observation": obs},
                                ALIASES, step_index=1, turn_index=0)
        self.assertEqual(state["completed_actions"], ["CALCULATE"])
        self.assertEqual(state["recent_observations"], [state["last_observation"]])
        self.assertEqual(state["last_observation"]["data"], {"result": 12})


def _wrote(path):
    return {"action": "write_file", "data": {"success": True, "result": f"Successfully wrote 9 characters to {path}",
                                             "tool": "write_file", "source": "builtin"}, "status": "success", "error": None}


class WriteFileTests(unittest.TestCase):
    def test_only_writes_of_new_files_become_file_create(self) -> None:
        listing = {"action": "list_directory", "data": {"success": True, "result": ["session_state.json (964 bytes)"],
                                                        "tool": "list_directory", "source": "builtin"},
                   "status": "success", "error": None}
        read = {"action": "read_file", "data": {"success": True, "result": "x", "tool": "read_file", "source": "builtin"},
                "status": "success", "error": None}
        command = {"action": "run_command", "data": {"success": True, "result": "ok", "tool": "run_command",
                                                     "source": "builtin"}, "status": "success", "error": None}
        arguments = {
            ("e1", 0): ("list_directory", {"dir_path": "."}),
            ("e1", 1): ("write_file", {"file_path": "index.html", "content": "a"}),
            ("e1", 2): ("write_file", {"file_path": "./index.html", "content": "b"}),
            ("e1", 3): ("read_file", {"file_path": "app.js"}),
            ("e1", 4): ("write_file", {"file_path": "app.js", "content": "c"}),
            ("e1", 5): ("write_file", {"file_path": "css/site.css", "content": "d"}),
            ("e1", 6): ("run_command", {"command": "npm create vite@latest app"}),
            ("e1", 7): ("write_file", {"file_path": "app/src/App.jsx", "content": "e"}),
        }
        episodes = [("e1", [listing, _wrote("index.html"), _wrote("index.html"), read, _wrote("app.js"),
                            _wrote("css/site.css"), command, _wrote("app/src/App.jsx")])]
        self.assertEqual(new_file_write_steps(arguments, episodes), {("e1", 1), ("e1", 5)})

    def test_state_maps_write_file_only_at_created_steps(self) -> None:
        obs = _wrote("index.html")
        state = normalize_state({"completed_actions": ["list_directory", "write_file"], "last_observation": obs},
                                ALIASES, step_index=2, turn_index=0,
                                step_aliases=lambda step: WRITE_AS_CREATE if step == 1 else {})
        self.assertEqual(state["completed_actions"], ["FILE_LIST", "FILE_CREATE"])
        self.assertEqual(state["last_observation"]["action"], "FILE_CREATE")
        with self.assertRaises(Reject):
            normalize_state({"completed_actions": ["write_file"], "last_observation": obs},
                            ALIASES, step_index=1, turn_index=0)

    def test_file_create_arguments_and_length_cap(self) -> None:
        self.assertEqual(normalize_arguments("FILE_CREATE", {"file_path": "a.md", "content": "x"}),
                         {"path": "a.md", "content": "x"})
        with self.assertRaises(Reject) as caught:
            normalize_arguments("FILE_CREATE", {"file_path": "a.html", "content": "x" * (MAX_CREATE_CONTENT + 1)})
        self.assertEqual(caught.exception.reason, "file_create_content_too_long")


class ResponseTests(unittest.TestCase):
    def test_tags_and_boxed_tails_are_removed(self) -> None:
        self.assertEqual(clean_response("<answer>Sonuç 42.</answer>"), "Sonuç 42.")
        self.assertEqual(clean_response("O kutular için 9.0 dolar ödüyor. Bu nedenle, boxed9."),
                         "O kutular için 9.0 dolar ödüyor.")

    def test_english_answers_are_rejected(self) -> None:
        with self.assertRaises(Reject):
            clean_response("I've created a minimalist landing page for you and this is the result of it.")

    def test_corrupted_numbers_are_rejected(self) -> None:
        check_numbers_against_results("Sonuç 0.9115370438628824'tür.", [0.9115370438628824], "")
        check_numbers_against_results("Yaklaşık 0,91 eder.", [0.9115370438628824], "")
        check_numbers_against_results("960.000 dolar kazanıyor.", [960000], "")
        check_numbers_against_results("Son toplam 23,248 $.", [23248.0], "")
        with self.assertRaises(Reject):
            check_numbers_against_results("Sonuç 1.183397668339766836'dır.", [1.1833976833976836], "")

    def test_calculations_are_recomputed(self) -> None:
        check_calculation("15*0.6", 9.0, "30 kutu alır, 1 alana 1 bedava.")
        with self.assertRaises(Reject) as caught:
            check_calculation("22479418+867", 22480285, "2247918 + 867 = ?", operands_from_prompt=True)
        self.assertEqual(caught.exception.reason, "calculate_operand_not_in_prompt")
        check_calculation("577-66111", -65534, "577 - 66111 = ?", operands_from_prompt=True)
        # tool-math uses intermediate results; only calculator rows are checked.
        check_calculation("1001 * 1001", 1002001, "1000 2 + 1,001 2 + 1,002 2 = ?")
        with self.assertRaises(Reject) as caught:
            check_calculation("2+2", 5, "iki artı iki")
        self.assertEqual(caught.exception.reason, "calculate_result_mismatch")


if __name__ == "__main__":
    unittest.main()
