from __future__ import annotations

import unittest
from unittest.mock import patch

from runtime import GenerationConfig, OllamaBackend


class OllamaBackendTests(unittest.TestCase):
    def test_thinking_is_disabled_by_default(self) -> None:
        captured = {}

        def fake_post(url, payload, *, timeout, headers=None):
            captured["url"] = url
            captured["payload"] = payload
            captured["timeout"] = timeout
            return {"message": {"role": "assistant", "content": "SEND_MAIL"}}

        backend = OllamaBackend("qwen3:1.7b")
        with patch("runtime.ollama_backend._post_json", side_effect=fake_post):
            result = backend.generate(
                [
                    {"role": "system", "content": "Choose one action."},
                    {"role": "user", "content": "mail gonder"},
                ],
                config=GenerationConfig(max_tokens=16),
            )

        self.assertEqual(result, "SEND_MAIL")
        self.assertIs(captured["payload"]["think"], False)
        self.assertEqual(captured["payload"]["options"]["num_predict"], 16)

    def test_empty_content_has_useful_diagnostic(self) -> None:
        def fake_post(url, payload, *, timeout, headers=None):
            return {
                "done_reason": "length",
                "message": {"role": "assistant", "content": "", "thinking": "abc"},
            }

        backend = OllamaBackend("qwen3:1.7b")
        with patch("runtime.ollama_backend._post_json", side_effect=fake_post):
            with self.assertRaisesRegex(RuntimeError, "empty message.content"):
                backend.generate([{"role": "user", "content": "test"}])


if __name__ == "__main__":
    unittest.main()
