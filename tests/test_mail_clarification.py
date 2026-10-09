"""Regression tests for clarification before SEND_MAIL execution."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import cli
from main import AgentPipeline
from runtime.mail_clarification import (
    clarification_question,
    is_mail_send_request,
    missing_mail_details,
)


class FakeYasama:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, *, user_prompt, conversation_history, state):
        self.calls += 1
        return "FINISH" if state["completed_actions"] else "SEND_MAIL"


class FakeYurutme:
    def __init__(self, answer=None) -> None:
        self.calls = 0
        self.answer = answer or {
            "to": "demo@example.org",
            "subject": "Tanışma",
            "body": "Merhaba, tanışmak isterim.",
        }

    def run(self, **kwargs):
        self.calls += 1
        return self.answer


class FakeYargi:
    def run(self, **kwargs):
        return "İşlem tamamlandı."


class MailClarificationTests(unittest.TestCase):
    def make_pipeline(self, *, yurutme=None, executor=None, traces=None):
        yurutme = yurutme or FakeYurutme()
        observations = []
        def fake_executor(action, arguments):
            observations.append((action, arguments))
            return {
                "action": action,
                "data": {"to": arguments["to"]},
                "status": "success",
                "error": None,
            }
        pipeline = AgentPipeline(
            yasama=FakeYasama(),
            yurutme=yurutme,
            yargi=FakeYargi(),
            tool_registry={},
            executor=executor or fake_executor,
            trace_handler=(lambda name, payload: traces.append((name, payload)))
            if traces is not None else None,
        )
        return pipeline, yurutme, observations

    def test_vague_mail_request_asks_topic_without_calling_yurutme(self):
        traces = []
        pipeline, yurutme, observations = self.make_pipeline(traces=traces)
        answer = pipeline.run("Mail yollamak istiyorum")
        self.assertEqual(answer, "Hangi konuda mail yollamak istiyorsunuz?")
        self.assertEqual(pipeline.last_action, "ASK_CLARIFICATION")
        self.assertEqual(yurutme.calls, 0)
        self.assertEqual(observations, [])
        self.assertEqual(traces[0][1]["model_action"], "SEND_MAIL")

    def test_explicit_mail_request_can_reach_executor(self):
        pipeline, yurutme, observations = self.make_pipeline()
        result = pipeline.run(
            'demo@example.org adresine "Tanışma" başlıklı kısa bir selamlama maili gönder.'
        )
        self.assertEqual(result, "İşlem tamamlandı.")
        self.assertEqual(yurutme.calls, 1)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0][0], "SEND_MAIL")

    def test_successful_mail_is_not_sent_twice(self):
        pipeline, _, observations = self.make_pipeline()
        pipeline.yasama.run = lambda **kwargs: "SEND_MAIL"
        result = pipeline.run(
            'demo@example.org adresine "Tanışma" başlıklı kısa bir selamlama maili gönder.'
        )
        self.assertEqual(result, "İşlem tamamlandı.")
        self.assertEqual(len(observations), 1)

    def test_invalid_mail_arguments_never_reach_executor(self):
        yurutme = FakeYurutme(answer={"role": "assistant", "content": "Bilgileri yazın."})
        pipeline, _, observations = self.make_pipeline(yurutme=yurutme)
        with self.assertRaisesRegex(ValueError, "unexpected argument fields"):
            pipeline.run('demo@example.org adresine "Tanışma" başlıklı kısa bir selamlama maili gönder.')
        self.assertEqual(observations, [])

    def test_clarification_fields_and_intent(self):
        self.assertTrue(is_mail_send_request("Mail yollamak istiyorum"))
        self.assertTrue(is_mail_send_request("Maili gönder"))
        self.assertEqual(
            missing_mail_details("Mail yollamak istiyorum"),
            ("to", "subject", "body"),
        )
        self.assertEqual(
            clarification_question(("to",)),
            "Maili kime göndermek istiyorsunuz? E-posta adresini yazar mısınız?",
        )
        self.assertFalse(is_mail_send_request("Mail nedir?"))

    def test_chat_preserves_pending_mail_intent_until_fields_supplied(self):
        pipeline, _, observations = self.make_pipeline()
        args = cli.build_parser().parse_args(["chat"])
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("cli.build_runtime", return_value=(pipeline, [])), patch(
            "builtins.input",
            side_effect=[
                "Mail yollamak istiyorum",
                "Tanışma",
                "demo@example.org",
                "Merhaba, tanışmak isterim.",
                "/exit",
            ],
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.cmd_chat(args)
        self.assertEqual(code, 0)
        self.assertIn("Hangi konuda mail yollamak istiyorsunuz?", stdout.getvalue())
        self.assertIn("Maili kime göndermek istiyorsunuz?", stdout.getvalue())
        self.assertIn("Mailin içeriğinde ne yazmamı istersiniz?", stdout.getvalue())
        self.assertIn("İşlem tamamlandı.", stdout.getvalue())
        self.assertEqual(len(observations), 1)
        self.assertNotIn("[error]", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
