from __future__ import annotations

import unittest

from cli_entry import _model_test_backend_config, build_model_test_parser


class ModelTestCliTests(unittest.TestCase):
    def test_ollama_model_test_defaults(self) -> None:
        parser = build_model_test_parser()
        args = parser.parse_args(["ollama", "qwen2.5:7b"])
        config = _model_test_backend_config(args)

        self.assertEqual(config["backend"], "ollama")
        self.assertEqual(config["model"], "qwen2.5:7b")
        self.assertEqual(config["base_url"], "http://127.0.0.1:11434")
        self.assertEqual(args.max_tokens, 128)
        self.assertEqual(args.temperature, 0.0)

    def test_llamacpp_base_url_override(self) -> None:
        parser = build_model_test_parser()
        args = parser.parse_args(
            [
                "llamacpp",
                "router",
                "--base-url",
                "http://127.0.0.1:9000",
                "--api-key",
                "secret",
            ]
        )
        config = _model_test_backend_config(args)

        self.assertEqual(config["base_url"], "http://127.0.0.1:9000")
        self.assertEqual(config["api_key"], "secret")

    def test_libmercan_options_are_forwarded(self) -> None:
        parser = build_model_test_parser()
        args = parser.parse_args(
            [
                "libmercan",
                "/models/model.mercan",
                "--library",
                "/opt/mercan/lib/libmercan.so",
                "--threads",
                "8",
                "--gpu-layers",
                "-1",
                "--n-ctx",
                "8192",
                "--plugin",
                "/plugins/arch.so",
            ]
        )
        config = _model_test_backend_config(args)

        self.assertEqual(config["library"], "/opt/mercan/lib/libmercan.so")
        self.assertEqual(config["threads"], 8)
        self.assertEqual(config["gpu_layers"], -1)
        self.assertEqual(config["n_ctx"], 8192)
        self.assertEqual(config["plugins"], ["/plugins/arch.so"])


if __name__ == "__main__":
    unittest.main()
