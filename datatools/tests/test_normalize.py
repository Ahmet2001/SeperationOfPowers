import json
import unittest

from datatools.normalize import (
    Reject,
    calculator_expression,
    check_calculation,
    check_numbers_against_results,
    clean_response,
    load_aliases,
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
        with self.assertRaises(Reject):
            check_numbers_against_results("Sonuç 1.183397668339766836'dır.", [1.1833976833976836], "")

    def test_calculations_are_recomputed(self) -> None:
        check_calculation("15*0.6", 9.0, "30 kutu alır, 1 alana 1 bedava.")
        with self.assertRaises(Reject) as caught:
            check_calculation("22479418+867", 22480285, "2247918 + 867 = ?")
        self.assertEqual(caught.exception.reason, "calculate_operand_not_in_prompt")
        with self.assertRaises(Reject) as caught:
            check_calculation("2+2", 5, "iki artı iki")
        self.assertEqual(caught.exception.reason, "calculate_result_mismatch")


if __name__ == "__main__":
    unittest.main()
