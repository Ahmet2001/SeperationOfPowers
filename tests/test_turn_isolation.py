"""General turn-isolation regressions (no Ollama, Torch or speaker required)."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import cli
from main import AgentPipeline


class FinishOnlyRouter:
    def run(self, **kwargs):
        return "FINISH"


class EchoJudge:
    def __init__(self):
        self.calls = []

    def run(self, *, user_prompt, conversation_history, observations):
        self.calls.append((user_prompt, list(conversation_history), list(observations)))
        return f"Cevap: {user_prompt}"


class NoToolRunner:
    def run(self, **kwargs):
        raise AssertionError("An ordinary response must not call Yurutme")


class TurnIsolationTests(unittest.TestCase):
    def test_finish_without_observation_becomes_respond(self):
        judge = EchoJudge()
        trace = []
        p = AgentPipeline(
            yasama=FinishOnlyRouter(),
            yurutme=NoToolRunner(),
            yargi=judge,
            tool_registry={},
            executor=lambda *_: self.fail("no executor call permitted"),
            trace_handler=lambda event, payload: trace.append((event, payload)),
        )
        for question in ("Hangi araçlara sahipsin?", "Sen hangi modelsin?", "Nasılsın?"):
            self.assertEqual(p.run(question), f"Cevap: {question}")
            self.assertEqual(p.last_action, "RESPOND")
        self.assertEqual(
            [entry[1]["model_action"] for entry in trace if entry[0] == "yasama"],
            ["FINISH", "FINISH", "FINISH"],
        )
        self.assertTrue(all(not call[2] for call in judge.calls))

    def test_mail_draft_is_not_chat_history(self):
        class ControlledPipeline:
            def __init__(self):
                self.last_action = None
                self.calls = []

            def run(self, text, conversation_history=None):
                self.calls.append((text, list(conversation_history or [])))
                if text.startswith("Mail yollamak istiyorum"):
                    self.last_action = "ASK_CLARIFICATION"
                    return "Hangi konuda mail yollamak istiyorsunuz?"
                self.last_action = "RESPOND"
                return f"Yanıt: {text}"

        pipeline = ControlledPipeline()
        args = cli.build_parser().parse_args(["chat"])
        with patch("cli.build_runtime", return_value=(pipeline, [])), patch(
            "builtins.input", side_effect=[
                "Mail yollamak istiyorum",
                "Tanışma",
                "vazgeçtim",
                "Hangi araçlara sahipsin?",
                "Sen hangi modelsin?",
                "/exit",
            ],
        ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = cli.cmd_chat(args)
        self.assertEqual(result, 0)
        self.assertEqual(len(pipeline.calls), 4)
        self.assertEqual(pipeline.calls[2][0], "Hangi araçlara sahipsin?")
        self.assertEqual(pipeline.calls[2][1], [])
        self.assertEqual(pipeline.calls[3][0], "Sen hangi modelsin?")
        self.assertEqual(
            pipeline.calls[3][1],
            [
                {"role": "user", "content": "Hangi araçlara sahipsin?"},
                {"role": "assistant", "content": "Yanıt: Hangi araçlara sahipsin?"},
            ],
        )

    def test_final_answer_after_observation_keeps_finish(self):
        class ToolThenFinish:
            def run(self, *, state, **kwargs):
                return "FINISH" if state["completed_actions"] else "FILE_LIST"

        class FileArgs:
            def run(self, **kwargs):
                return {"path": ".", "pattern": "*.py", "recursive": False}

        observation = {
            "action": "FILE_LIST",
            "data": {"path": ".", "pattern": "*.py", "recursive": False, "entries": ["test.py"]},
            "status": "success",
            "error": None,
        }
        judge = EchoJudge()
        p = AgentPipeline(
            yasama=ToolThenFinish(),
            yurutme=FileArgs(),
            yargi=judge,
            tool_registry={},
            executor=lambda *_: self.fail("built-in tool should be used"),
        )
        with patch("main.execute_builtin_local", return_value=observation) as execute:
            result = p.run("Python dosyalarını listele")
        self.assertEqual(result, "Cevap: Python dosyalarını listele")
        self.assertEqual(p.last_action, "FINISH")
        self.assertEqual(judge.calls[0][2], [observation])
        execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
