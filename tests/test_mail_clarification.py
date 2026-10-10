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
    is_mail_cancel_request,
    is_mail_send_request,
    is_simple_chat_request,
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

    def test_stale_send_mail_from_history_cannot_hijack_smalltalk(self):
        traces = []
        pipeline, yurutme, observations = self.make_pipeline(traces=traces)

        class GreetingJudge:
            def run(self, *, user_prompt, conversation_history, observations):
                assert user_prompt == "Nasılsın"
                assert conversation_history == []  # No stale mail context.
                assert observations == []
                return "İyiyim, teşekkürler."

        pipeline.yargi = GreetingJudge()
        answer = pipeline.run(
            "Nasılsın",
            conversation_history=[
                {"role": "user", "content": "Mail yollamak istiyorum"},
                {"role": "assistant", "content": "Hangi konuda mail yollamak istiyorsunuz?"},
            ],
        )
        self.assertEqual(answer, "İyiyim, teşekkürler.")
        self.assertEqual(pipeline.last_action, "RESPOND")
        self.assertEqual(traces[0][1]["model_action"], "SEND_MAIL")
        self.assertEqual(yurutme.calls, 0)
        self.assertEqual(observations, [])

    def test_stale_mail_action_is_replanned_for_new_file_search(self):
        from main import AgentPipeline

        class ContextSensitiveRouter:
            def __init__(self):
                self.histories = []

            def run(self, *, user_prompt, conversation_history, state):
                self.histories.append(list(conversation_history))
                if state["completed_actions"]:
                    return "FINISH"
                return "SEND_MAIL" if conversation_history else "FILE_SEARCH"

        class SearchArgs:
            def run(self, **kwargs):
                self.action = kwargs["action"]
                return {
                    "query": "OllamaBackend", "path": ".", "pattern": "*.py",
                    "recursive": True, "max_results": 5
                }

        router = ContextSensitiveRouter()
        args = SearchArgs()
        pipeline = AgentPipeline(
            yasama=router,
            yurutme=args,
            yargi=FakeYargi(),
            tool_registry={},
            executor=lambda *_: self.fail("external executor should not be used"),
        )
        observation = {
            "action": "FILE_SEARCH",
            "data": {"matches": [{"path": "main.py", "line": 1, "text": "OllamaBackend"}]},
            "status": "success",
            "error": None,
        }
        with patch("main.execute_builtin_local", return_value=observation) as execute:
            result = pipeline.run(
                "Projede OllamaBackend geçen Python dosyalarını bul.",
                conversation_history=[
                    {"role": "user", "content": "Mail yollamak istiyorum"},
                    {"role": "assistant", "content": "Hangi konuda mail yollamak istiyorsunuz?"},
                ],
            )
        self.assertEqual(result, "İşlem tamamlandı.")
        self.assertEqual(args.action, "FILE_SEARCH")
        execute.assert_called_once()
        self.assertTrue(router.histories[0])
        self.assertEqual(router.histories[1], [])

    def test_refused_mail_is_never_treated_as_send(self):
        self.assertFalse(is_mail_send_request("Mail yollamak istemiyorum"))
        self.assertFalse(is_mail_send_request("Maili gönderme"))
        self.assertFalse(is_mail_send_request("Mail adresimi öğrenmek istiyorum"))
        self.assertTrue(is_mail_send_request("Mail göndermek istiyorum"))
        self.assertFalse(is_mail_cancel_request("Toplantı iptal mi?"))
        self.assertTrue(is_mail_cancel_request("mail yollamak istemiyorum"))
        self.assertTrue(is_mail_cancel_request("Maili gönderme"))
        self.assertFalse(is_mail_cancel_request("Mail yollamak istiyorum"))
        self.assertTrue(is_simple_chat_request("Nasılsın?"))

        pipeline, yurutme, observations = self.make_pipeline()
        answer = pipeline.run(
            "Mail yollamak istemiyorum",
            conversation_history=[{"role": "user", "content": "Mail yollamak istiyorum"}],
        )
        self.assertEqual(answer, "Tamam, mail göndermeyeceğim.")
        self.assertEqual(pipeline.last_action, "RESPOND")
        self.assertEqual(yurutme.calls, 0)
        self.assertEqual(observations, [])

    def test_chat_continues_normally_after_missing_mail_executor(self):
        pipeline, yurutme, _ = self.make_pipeline(
            executor=lambda action, args: (_ for _ in ()).throw(
                RuntimeError("no mail executor")
            )
        )
        args = cli.build_parser().parse_args(["chat"])
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("cli.build_runtime", return_value=(pipeline, [])), patch(
            "builtins.input",
            side_effect=[
                "Mail yollamak istiyorum",
                "Tanışma",
                "demo@example.org",
                "Merhaba, tanışmak isterim.",
                "Nasılsın",
                "mail yollamak istemiyorum",
                "/exit",
            ],
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.cmd_chat(args)
        self.assertEqual(code, 0)
        self.assertEqual(
            stdout.getvalue().count("Hangi konuda mail yollamak istiyorsunuz?"),
            1,
        )
        self.assertIn("Tamam, mail göndermeyeceğim.", stdout.getvalue())
        self.assertIn("[error] no mail executor", stderr.getvalue())
        self.assertEqual(yurutme.calls, 1)

    def test_chat_cancellation_during_mail_does_not_send(self):
        pipeline, yurutme, observations = self.make_pipeline()
        args = cli.build_parser().parse_args(["chat"])
        stdout = io.StringIO()
        with patch("cli.build_runtime", return_value=(pipeline, [])), patch(
            "builtins.input",
            side_effect=[
                "Mail yollamak istiyorum",
                "mail yollamak istemiyorum",
                "Nasılsın",
                "/exit",
            ],
        ), redirect_stdout(stdout):
            code = cli.cmd_chat(args)
        self.assertEqual(code, 0)
        self.assertIn("Tamam, mail göndermeyeceğim.", stdout.getvalue())
        self.assertEqual(
            stdout.getvalue().count("Hangi konuda mail yollamak istiyorsunuz?"),
            1,
        )
        self.assertEqual(yurutme.calls, 0)
        self.assertEqual(observations, [])

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
