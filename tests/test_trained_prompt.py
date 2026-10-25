"""A fine-tuned role sees exactly its training system prompt and no few-shot turns."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import cli
from runtime.local_tools import BUILTIN_TOOL_REGISTRY
from runtime.tool_validation import public_tool_schema
from yargi import yargi as yargi_module
from yargi.yargi import YargiRuntime
from yasama import yasama as yasama_module
from yasama.yasama import YasamaRuntime
from yurutme import yurutme as yurutme_module
from yurutme.yurutme import YurutmeRuntime

STATE = {"completed_actions": [], "last_observation": None, "recent_observations": [],
         "step_index": 0, "turn_index": 0}


class FakeBackend:
    def __init__(self, reply):
        self.reply = reply
        self.last_messages = None

    def generate(self, messages, *, config=None):
        self.last_messages = messages
        return self.reply


class TrainedPromptTests(unittest.TestCase):
    def test_trained_roles_send_system_prompt_and_request_only(self):
        backend = FakeBackend("RESPOND")
        YasamaRuntime(backend=backend, trained=True).run(user_prompt="Merhaba", conversation_history=[], state=STATE)
        self.assertEqual(backend.last_messages[0], yasama_module.system_message())
        self.assertEqual(len(backend.last_messages), 2)

        backend = FakeBackend('{"expression": "2+2"}')
        tool = BUILTIN_TOOL_REGISTRY["CALCULATE"]
        YurutmeRuntime(backend=backend, trained=True).run(
            user_prompt="2+2 kaç?", conversation_history=[], state=STATE, action="CALCULATE",
            tool_schema=public_tool_schema(tool), placeholder=tool.get("placeholder"))
        self.assertEqual(backend.last_messages[0], yurutme_module.system_message())
        self.assertEqual(len(backend.last_messages), 2)

        backend = FakeBackend("Merhaba!")
        yargi = YargiRuntime(backend=backend, trained=True)
        yargi.set_runtime_info({"roles": {"yargi": {"backend": "libmercan", "model": "/m.mercan"}}})
        yargi.run(user_prompt="Merhaba", conversation_history=[], observations=[])
        self.assertEqual(backend.last_messages[0], yargi_module.system_message())
        self.assertEqual(len(backend.last_messages), 2)

    def test_untrained_roles_keep_few_shot_turns(self):
        backend = FakeBackend("RESPOND")
        YasamaRuntime(backend=backend).run(user_prompt="Merhaba", conversation_history=[], state=STATE)
        self.assertEqual(backend.last_messages[0]["content"], yasama_module.YASAMA_SYSTEM_PROMPT)
        self.assertGreater(len(backend.last_messages), 2)

    def test_yasama_prompt_lists_every_action(self):
        for action in yasama_module.ACTION_ORDER:
            self.assertIn(f"- {action}: ", yasama_module.TRAINED_SYSTEM_PROMPT)

    def test_cli_prompt_style(self):
        config = json.loads(json.dumps(cli.DEFAULT_CONFIG))
        config["roles"]["yargi"]["prompt_style"] = "trained"
        with patch.dict(os.environ, {}, clear=True):
            pipeline, _ = cli.build_runtime(config)
        self.assertTrue(pipeline.yargi._trained)
        self.assertFalse(pipeline.yasama._trained)
        config["roles"]["yasama"]["prompt_style"] = "fine"
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            cli.build_runtime(config)

    def test_cli_generation_follows_prompt_style_and_overrides(self):
        config = json.loads(json.dumps(cli.DEFAULT_CONFIG))
        config["roles"]["yurutme"]["prompt_style"] = "trained"
        config["roles"]["yargi"]["generation"] = {"max_tokens": 100}
        with patch.dict(os.environ, {}, clear=True):
            pipeline, _ = cli.build_runtime(config)
        self.assertEqual(pipeline.yurutme._generation_config, yurutme_module.default_generation_config(trained=True))
        self.assertEqual(pipeline.yurutme._generation_config.max_tokens, yurutme_module.TRAINED_MAX_TOKENS)
        self.assertEqual(pipeline.yargi._generation_config.max_tokens, 100)
        self.assertEqual(pipeline.yargi._generation_config.temperature,
                         yargi_module.default_generation_config().temperature)
        config["roles"]["yargi"]["generation"] = {"max_token": 100}
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            cli.build_runtime(config)


if __name__ == "__main__":
    unittest.main()
