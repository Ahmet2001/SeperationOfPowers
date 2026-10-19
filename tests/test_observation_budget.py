"""Each role sees the observation detail it needs, untruncated; missing executors do not crash."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import cli
from main import AgentPipeline
from runtime.serialization import (
    serialize_yargi_input,
    serialize_yasama_input,
    serialize_yurutme_input,
    yasama_state,
    yurutme_state,
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


HISTORY = [{"role": "user", "content": "docs klasörünü listele"},
           {"role": "assistant", "content": "docs klasöründe api.md var."}]


class RoleViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.content = "z" * 10_000
        self.observation = read_observation(self.content)
        self.state = {"completed_actions": ["FILE_READ"], "last_observation": self.observation,
                      "recent_observations": [self.observation], "step_index": 1}

    def test_yasama_sees_outcomes_without_content(self) -> None:
        view = yasama_state(self.state)
        self.assertEqual(view["recent_observations"],
                         [{"action": "FILE_READ", "status": "success", "error": None}])
        self.assertEqual(view["completed_actions"], ["FILE_READ"])
        prompt = serialize_yasama_input(user_prompt="oku", conversation_history=[], state=self.state)
        self.assertNotIn("zzz", prompt)

    def test_yurutme_sees_full_content_once(self) -> None:
        view = yurutme_state(self.state)
        self.assertNotIn("last_observation", view)
        self.assertEqual(view["recent_observations"], [self.observation])
        prompt = serialize_yurutme_input(
            user_prompt="oku", conversation_history=[], state=self.state,
            action="FILE_EDIT", tool_schema={}, placeholder=None,
        )
        self.assertEqual(prompt.count(self.content), 1)

    def test_yargi_sees_full_content(self) -> None:
        prompt = serialize_yargi_input(user_prompt="oku", conversation_history=[],
                                       observations=[self.observation])
        self.assertIn(self.content, prompt)
        self.assertNotIn("karakter)", prompt)

    def test_distinct_last_observation_is_kept(self) -> None:
        other = read_observation("farklı")
        state = {**self.state, "last_observation": other}
        self.assertEqual(yurutme_state(state)["last_observation"], other)

    def test_views_do_not_modify_the_pipeline_state(self) -> None:
        yasama_state(self.state)
        yurutme_state(self.state)
        self.assertIn("last_observation", self.state)
        self.assertEqual(self.state["recent_observations"][0]["data"]["content"], self.content)


class YasamaHistoryTests(unittest.TestCase):
    def test_first_step_carries_conversation_history(self) -> None:
        state = {"completed_actions": [], "last_observation": None, "recent_observations": []}
        prompt = serialize_yasama_input(user_prompt="aynısını tests için yap",
                                        conversation_history=HISTORY, state=state)
        self.assertIn("CONVERSATION_HISTORY:", prompt)

    def test_later_steps_carry_only_the_step_chain(self) -> None:
        observation = read_observation("içerik")
        state = {"completed_actions": ["FILE_LIST"], "last_observation": observation,
                 "recent_observations": [observation]}
        prompt = serialize_yasama_input(user_prompt="aynısını tests için yap",
                                        conversation_history=HISTORY, state=state)
        self.assertNotIn("CONVERSATION_HISTORY:", prompt)
        self.assertIn('"completed_actions"', prompt)

    def test_yurutme_keeps_history_after_tools_run(self) -> None:
        observation = read_observation("içerik")
        state = {"completed_actions": ["FILE_LIST"], "last_observation": observation,
                 "recent_observations": [observation]}
        prompt = serialize_yurutme_input(
            user_prompt="aynısını tests için yap", conversation_history=HISTORY, state=state,
            action="FILE_LIST", tool_schema={}, placeholder=None,
        )
        self.assertIn("CONVERSATION_HISTORY:", prompt)


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


class ContextDefaultTests(unittest.TestCase):
    def build(self, extra: dict) -> dict:
        with patch("cli.LibMercanBackend") as backend:
            cli._build_backend("yargi", {"backend": "libmercan", "model": "/m.mercan", **extra})
        return backend.call_args.kwargs

    def test_libmercan_roles_default_to_8k_context(self) -> None:
        self.assertEqual(self.build({})["n_ctx"], cli.LIBMERCAN_DEFAULT_N_CTX)
        self.assertEqual(cli.LIBMERCAN_DEFAULT_N_CTX, 8192)

    def test_explicit_context_wins(self) -> None:
        self.assertEqual(self.build({"n_ctx": 16384})["n_ctx"], 16384)


if __name__ == "__main__":
    unittest.main()
