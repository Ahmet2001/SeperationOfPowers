"""Optional EMA Lightning speech integration tests (no downloads/audio hardware)."""

from __future__ import annotations

import io
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cli
from runtime import speech as speech_module
from runtime.speech import EMALightningSpeech, MODEL_REPO


class FakePipeline:
    last_action = "RESPOND"

    def run(self, prompt, conversation_history=None):
        return f"Yanıt: {prompt}"


class SpeechIntegrationTests(unittest.TestCase):
    def test_lazy_ema_model_is_shared_across_messages(self):
        speech = SimpleNamespace(audio="fake-pcm", sample_rate=48000)
        model = Mock()
        model.say.return_value = speech
        factory = Mock(return_value=model)
        playback = Mock()
        speaker = EMALightningSpeech(ema_factory=factory, playback=playback)
        self.assertEqual(MODEL_REPO, "canberkkkkkk/ema-lightning")
        factory.assert_not_called()
        speaker.speak("Merhaba!")
        speaker.speak("Nasılsın?")
        factory.assert_called_once_with()
        self.assertEqual(model.say.call_count, 2)
        playback.assert_any_call("fake-pcm", 48000)

    def test_default_ema_loader_selects_cpu_explicitly(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.version = SimpleNamespace(cuda=None, hip=None)
        fake_ema_module = types.ModuleType("ema_lightning")
        constructor = Mock(return_value=object())
        fake_ema_module.EMA = constructor

        with patch.dict(
            sys.modules, {"torch": fake_torch, "ema_lightning": fake_ema_module}
        ):
            speech_module._load_ema()

        constructor.assert_called_once_with(device="cpu")

    def test_default_ema_loader_rejects_cuda_enabled_torch(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.version = SimpleNamespace(cuda="12.8", hip=None)
        fake_ema_module = types.ModuleType("ema_lightning")
        constructor = Mock()
        fake_ema_module.EMA = constructor

        with patch.dict(
            sys.modules, {"torch": fake_torch, "ema_lightning": fake_ema_module}
        ):
            with self.assertRaisesRegex(RuntimeError, "CPU-only PyTorch"):
                speech_module._load_ema()

        constructor.assert_not_called()

    def test_default_ema_loader_rejects_rocm_torch(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.version = SimpleNamespace(cuda=None, hip="6.3")
        fake_ema_module = types.ModuleType("ema_lightning")
        constructor = Mock()
        fake_ema_module.EMA = constructor

        with patch.dict(
            sys.modules, {"torch": fake_torch, "ema_lightning": fake_ema_module}
        ):
            with self.assertRaisesRegex(RuntimeError, "CPU-only PyTorch"):
                speech_module._load_ema()

        constructor.assert_not_called()

    def test_empty_answer_does_not_load_tts(self):
        factory = Mock()
        speaker = EMALightningSpeech(ema_factory=factory, playback=Mock())
        speaker.speak("   ")
        factory.assert_not_called()

    def test_speech_flag_is_opt_in(self):
        parser = cli.build_parser()
        silent = cli._resolve_config(parser.parse_args(["chat"]))
        voiced = cli._resolve_config(parser.parse_args(["chat", "--speech"]))
        one_shot = cli._resolve_config(
            parser.parse_args(["run", "--speech", "Merhaba"])
        )
        self.assertFalse(silent["pipeline"]["speech"])
        self.assertTrue(voiced["pipeline"]["speech"])
        self.assertTrue(one_shot["pipeline"]["speech"])

    def test_no_tts_construction_without_speech_flag(self):
        args = cli.build_parser().parse_args(["run", "Merhaba"])
        stdout = io.StringIO()
        with patch("cli.build_runtime", return_value=(FakePipeline(), [])), patch(
            "cli.EMALightningSpeech"
        ) as constructor, redirect_stdout(stdout):
            result = cli.cmd_run(args)
        self.assertEqual(result, 0)
        self.assertIn("Yanıt: Merhaba", stdout.getvalue())
        constructor.assert_not_called()

    def test_run_speaks_final_answer_not_tool_arguments(self):
        args = cli.build_parser().parse_args(["run", "--speech", "Merhaba"])
        voiced = Mock()
        stdout = io.StringIO()
        with patch("cli.build_runtime", return_value=(FakePipeline(), [])), patch(
            "cli.EMALightningSpeech", return_value=voiced
        ), redirect_stdout(stdout):
            result = cli.cmd_run(args)
        self.assertEqual(result, 0)
        voiced.speak.assert_called_once_with("Yanıt: Merhaba")
        self.assertIn("Yanıt: Merhaba", stdout.getvalue())

    def test_chat_reuses_voice_for_multiple_turns(self):
        args = cli.build_parser().parse_args(["chat", "--speech"])
        voiced = Mock()
        stdout = io.StringIO()
        with patch("cli.build_runtime", return_value=(FakePipeline(), [])), patch(
            "cli.EMALightningSpeech", return_value=voiced
        ), patch("builtins.input", side_effect=["Selam", "Nasılsın", "/exit"]), redirect_stdout(stdout):
            result = cli.cmd_chat(args)
        self.assertEqual(result, 0)
        self.assertEqual(
            [call.args[0] for call in voiced.speak.call_args_list],
            ["Yanıt: Selam", "Yanıt: Nasılsın"],
        )

    def test_audio_failure_does_not_lose_answer(self):
        args = cli.build_parser().parse_args(["run", "--speech", "Merhaba"])
        voiced = Mock()
        voiced.speak.side_effect = RuntimeError("Audio device unavailable")
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("cli.build_runtime", return_value=(FakePipeline(), [])), patch(
            "cli.EMALightningSpeech", return_value=voiced
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            result = cli.cmd_run(args)
        self.assertEqual(result, 0)
        self.assertIn("Yanıt: Merhaba", stdout.getvalue())
        self.assertIn("[speech:error] Audio device unavailable", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
