from __future__ import annotations

import json
import unittest

from runtime.local_tools import BUILTIN_TOOL_REGISTRY
from runtime.tool_validation import public_tool_schema
from training.eval_metrics import score_yargi_one, score_yasama, score_yurutme_one
from yargi.yargi import request_messages
from yurutme.yurutme import request_message

STATE = {"completed_actions": [], "last_observation": None, "recent_observations": [],
         "step_index": 0, "turn_index": 0}


def yurutme_request(action, prompt="istek"):
    tool = BUILTIN_TOOL_REGISTRY[action]
    return request_message(user_prompt=prompt, conversation_history=[], state=STATE, action=action,
                           tool_schema=public_tool_schema(tool), placeholder=tool.get("placeholder"))["content"]


def yargi_request(observations, prompt="istek"):
    return request_messages(user_prompt=prompt, conversation_history=[], observations=observations)[-1]["content"]


class YasamaTests(unittest.TestCase):
    def test_accuracy_and_confusions(self):
        report = score_yasama([{"target": "FILE_READ", "output": "FILE_READ"},
                               {"target": "FILE_READ", "output": "FILE_SEARCH"},
                               {"target": "RESPOND", "output": "RESPOND\n"},
                               {"target": "RESPOND", "output": "Merhaba"}])
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["top_confusions"], {"FILE_READ -> FILE_SEARCH": 1, "RESPOND -> Merhaba": 1})
        self.assertEqual(report["invalid_outputs"], 2)


class YurutmeTests(unittest.TestCase):
    def test_equivalent_calculations_are_correct(self):
        request = yurutme_request("CALCULATE")
        result = score_yurutme_one(request, '{"expression": "15*0.6"}', '{"expression": "0.6 * 15"}')
        self.assertEqual((result["schema_valid"], result["exact"], result["semantic"]), (True, False, True))
        wrong = score_yurutme_one(request, '{"expression": "15*0.6"}', '{"expression": "15/0.6"}')
        self.assertFalse(wrong["semantic"])

    def test_schema_and_fields(self):
        request = yurutme_request("FILE_LIST")
        good = score_yurutme_one(request, '{"path": "src", "pattern": "*.py"}', '{"path": "src", "pattern": "*.py"}')
        self.assertTrue(good["exact"] and good["semantic"])
        half = score_yurutme_one(request, '{"path": "src", "pattern": "*.py"}', '{"path": "src", "pattern": "*"}')
        self.assertEqual((half["semantic"], half["field_match"]), (False, 0.5))
        extra = score_yurutme_one(request, '{"path": "src"}', '{"path": "src", "colour": "red"}')
        self.assertFalse(extra["schema_valid"])
        broken = score_yurutme_one(request, '{"path": "src"}', 'path: src')
        self.assertFalse(broken["json"])

    def test_code_is_checked_for_syntax(self):
        request = yurutme_request("CODE_EXECUTE")
        result = score_yurutme_one(request, json.dumps({"code": "print(1)", "language": "python"}),
                                   json.dumps({"code": "print(2)", "language": "python"}))
        self.assertTrue(result["python_parses"])
        result = score_yurutme_one(request, json.dumps({"code": "print(1)", "language": "python"}),
                                   json.dumps({"code": "print(", "language": "python"}))
        self.assertFalse(result["python_parses"])


class YargiTests(unittest.TestCase):
    def test_numbers_must_come_from_the_observations(self):
        request = yargi_request([{"action": "CALCULATE", "data": {"result": 156.25}, "status": "success", "error": None}],
                                "1250 lirayı 8 kişiye bölersek?")
        self.assertEqual(score_yargi_one(request, "Kişi başı 156,25 lira.", "Kişi başı 156,25 lira.")["ungrounded_numbers"], [])
        self.assertEqual(score_yargi_one(request, "x", "Kişi başı 165 lira.")["ungrounded_numbers"], ["165"])

    def test_failures_must_be_admitted_and_json_not_leaked(self):
        request = yargi_request([{"action": "FILE_READ", "data": {}, "status": "error", "error": "No such file"}])
        self.assertTrue(score_yargi_one(request, "x", "Dosya bulunamadı.")["admits_failure"])
        self.assertFalse(score_yargi_one(request, "x", "Dosyanın içeriği şöyle: merhaba.")["admits_failure"])
        self.assertTrue(score_yargi_one(request, "x", '{"action": "FILE_READ"}')["json_leak"])


if __name__ == "__main__":
    unittest.main()
