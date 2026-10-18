"""Observations stay inside a 4k context and missing executors do not crash."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import cli
from main import AgentPipeline
from runtime.serialization import (
    STATE_OBSERVATION_LIMITS,
    YARGI_OBSERVATION_LIMITS,
    serialize_yargi_input,
    serialize_yasama_input,
    serialize_yurutme_input,
    summarize_state,
    summarize_value,
)
from yargi.yargi import YargiRuntime
from yasama.yasama import YasamaRuntime
from yurutme.yurutme import YurutmeRuntime


def read_observation(content: str) -> dict:
    return {
        "action": "FILE_READ",
        "data": {"path": "main.py", "content": content, "truncated": False},
        "status": "success",
        "error": None,
    }


class SummaryTests(unittest.TestCase):
    def test_long_strings_and_lists_are_shortened_with_a_count(self) -> None:
        summary = summarize_value({"text": "x" * 70, "items": list(range(5))}, (60, 3))
        self.assertEqual(summary["text"], "x" * 60 + "… (+10 karakter)")
        self.assertEqual(summary["items"], [0, 1, 2, "… (+2 öğe)"])

    def test_small_observations_are_unchanged(self) -> None:
        observation = {"action": "FILE_CREATE", "data": {"path": "notes.txt"},
                       "status": "success", "error": None}
        self.assertEqual(summarize_value(observation, STATE_OBSERVATION_LIMITS), observation)

    def test_state_keeps_fields_and_shortens_only_observations(self) -> None:
        observation = read_observation("y" * 10_000)
        state = {"completed_actions": ["FILE_READ"], "last_observation": observation,
                 "recent_observations": [observation], "step_index": 1}
        summary = summarize_state(state)
        self.assertEqual(set(summary), set(state))
        self.assertEqual(summary["completed_actions"], ["FILE_READ"])
        self.assertLess(len(summary["last_observation"]["data"]["content"]), 100)
        self.assertEqual(summary["recent_observations"][0]["status"], "success")
        # The pipeline's own state object is not modified.
        self.assertEqual(len(state["last_observation"]["data"]["content"]), 10_000)


class SerializerBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = read_observation("z" * 10_000)
        self.state = {"completed_actions": ["FILE_READ"], "last_observation": self.observation,
                      "recent_observations": [self.observation]}

    def test_yasama_and_yurutme_see_short_observations(self) -> None:
        yasama = serialize_yasama_input(user_prompt="oku", conversation_history=[], state=self.state)
        yurutme = serialize_yurutme_input(
            user_prompt="oku", conversation_history=[], state=self.state,
            action="FILE_READ", tool_schema={}, placeholder=None,
        )
        for prompt in (yasama, yurutme):
            self.assertLess(len(prompt), 2000)
            self.assertIn("+9940 karakter", prompt)

    def test_yargi_keeps_more_content_but_is_bounded(self) -> None:
        prompt = serialize_yargi_input(user_prompt="oku", conversation_history=[],
                                       observations=[self.observation])
        self.assertIn("z" * YARGI_OBSERVATION_LIMITS[0], prompt)
        self.assertIn(f"+{10_000 - YARGI_OBSERVATION_LIMITS[0]} karakter", prompt)


class MissingExecutorTests(unittest.TestCase):
    def make_pipeline(self, executor) -> tuple[AgentPipeline, list]:
        yargi_inputs: list = []
        pipeline = AgentPipeline(
            yasama=YasamaRuntime(lambda payload: "SHELL_EXECUTE"),
            yurutme=YurutmeRuntime(lambda payload: {"command": "git status"}),
            yargi=YargiRuntime(lambda payload: (
                yargi_inputs.append(payload["observations"]) or "Komut çalıştırılamadı."
            )),
            tool_registry={},
            executor=executor,
            max_steps=5,
        )
        return pipeline, yargi_inputs

    def test_missing_executor_is_reported_instead_of_raised(self) -> None:
        observation = cli._missing_executor("SHELL_EXECUTE", {"command": "git status"})
        self.assertEqual(observation["status"], "error")
        self.assertIn("yapılandırılmamış", observation["error"])

    def test_repeated_identical_call_runs_once_and_ends_with_yargi(self) -> None:
        calls: list = []

        def executor(action, arguments):
            calls.append((action, arguments))
            return cli._missing_executor(action, arguments)

        pipeline, yargi_inputs = self.make_pipeline(executor)
        # Yasama keeps choosing SHELL_EXECUTE; without the guard this would
        # run the same command until max_steps and raise.
        self.assertEqual(pipeline.run("git status komutunu çalıştır."), "Komut çalıştırılamadı.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(yargi_inputs[0][0]["status"], "error")


class RuntimeInfoBudgetTests(unittest.TestCase):
    def test_descriptions_only_for_tools_that_can_run(self) -> None:
        config = json.loads(json.dumps(cli.DEFAULT_CONFIG))
        with patch.dict(os.environ, {}, clear=True):
            pipeline, _ = cli.build_runtime(config)
        tools = pipeline.yargi._runtime_info["tools"]
        self.assertIn("description", tools["FILE_READ"])
        self.assertNotIn("description", tools["SEND_MAIL"])
        self.assertFalse(tools["SEND_MAIL"]["external_executor_configured"])


if __name__ == "__main__":
    unittest.main()
