import json
import unittest

from datatools.build_shared_data import Context, yargi_record, yurutme_record
from datatools.normalize import (
    MAX_RECORD_CHARS,
    Reject,
    check_record_length,
    load_aliases,
    open_tool_schema,
)

ALIASES = load_aliases(None)
EMPTY = {"completed_actions": [], "last_observation": None}


def _ctx():
    return Context(ALIASES, queries={}, calc_results={}, new_writes=set())


class OpenToolSchemaTests(unittest.TestCase):
    def test_both_source_forms_become_the_runtime_shape(self) -> None:
        openai_form = {"type": "function", "function": {
            "name": "get_weather", "description": "Hava durumu.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }}
        self.assertEqual(open_tool_schema(openai_form), {
            "description": "Hava durumu.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            "placeholder": {"city": None},
        })
        misplaced = {"name": "calculate_interest", "description": "Faiz.",
                     "parameters": {"type": "object", "properties": {"rate": {"type": "number"}}},
                     "required": ["rate"]}
        self.assertEqual(open_tool_schema(misplaced)["parameters"]["required"], ["rate"])

    def test_missing_or_list_schemas_are_rejected(self) -> None:
        for schema, reason in [(None, "tool_schema_missing"),
                               ({"name": "x", "parameters": [{"name": "q"}]}, "tool_schema_unsupported")]:
            with self.subTest(schema=schema), self.assertRaises(Reject) as caught:
                open_tool_schema(schema)
            self.assertEqual(caught.exception.reason, reason)

    def test_long_records_are_rejected(self) -> None:
        check_record_length([{"role": "user", "content": "x" * MAX_RECORD_CHARS}])
        with self.assertRaises(Reject):
            check_record_length([{"role": "user", "content": "x" * MAX_RECORD_CHARS},
                                 {"role": "assistant", "content": "y"}])


class OpenToolRecordTests(unittest.TestCase):
    def test_yurutme_fills_a_source_tool_with_its_own_schema(self) -> None:
        row = {
            "source_id": "h-1", "step_index": 0, "user_prompt": "Hue ampulleri ve Nest termostatı bağla.",
            "conversation_history": [], "state": EMPTY, "action": "initialize_smart_home_system",
            "placeholder": {"device_list": None},
            "tool_schema": {"type": "function", "function": {
                "name": "initialize_smart_home_system", "description": "Akıllı ev sistemini başlatır.",
                "parameters": {"type": "object", "properties": {"device_list": {"type": "array"}},
                               "required": ["device_list"]}}},
            "target_arguments": {"device_list": ["Philips Hue", "Nest"]},
        }
        label, messages, meta = yurutme_record(row, "combined_v2", _ctx())
        self.assertEqual((label, meta["tool"]), ("OPEN_TOOL", "initialize_smart_home_system"))
        self.assertIn("ACTION:\ninitialize_smart_home_system", messages[0]["content"])
        self.assertIn('"description": "Akıllı ev sistemini başlatır."', messages[0]["content"])
        self.assertEqual(json.loads(messages[1]["content"]), {"device_list": ["Philips Hue", "Nest"]})

        row["target_arguments"] = {"devices": ["Nest"]}
        with self.assertRaises(Reject) as caught:
            yurutme_record(row, "combined_v2", _ctx())
        self.assertTrue(caught.exception.reason.startswith("open_schema:"))

    def test_yargi_keeps_source_tool_observations(self) -> None:
        row = {
            "source_id": "g-1", "user_prompt": "Boyum 1.75, kilom 70; BMI'm kaç?", "conversation_history": [],
            "observations": [{"action": "calculate_bmi", "tool_call_id": "c", "data": {"bmi": 22.86},
                              "status": "success", "error": None}],
            "response": "Vücut kitle indeksiniz 22.86, normal aralıkta.",
        }
        label, messages, _ = yargi_record(row, "arac_kullanimi", _ctx())
        self.assertEqual(label, "OPEN_TOOL")
        self.assertIn('"action": "calculate_bmi"', messages[0]["content"])
        self.assertNotIn("tool_call_id", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
