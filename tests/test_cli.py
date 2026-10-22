from __future__ import annotations

import unittest

import cli


class CliConfigTests(unittest.TestCase):
    def test_run_parser_accepts_role_overrides(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "run",
                "merhaba",
                "--yasama-backend",
                "ollama",
                "--yasama-model",
                "router-v1",
                "--set",
                "roles.yurutme.threads=8",
            ]
        )
        config = cli._resolve_config(args)

        self.assertEqual(config["roles"]["yasama"]["model"], "router-v1")
        self.assertEqual(config["roles"]["yurutme"]["threads"], 8)

    def test_deep_set_can_override_backend_specific_values(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "show-config",
                "--set",
                "roles.yargi.backend=libmercan",
                "--set",
                "roles.yargi.model=/tmp/yargi.mercan",
                "--set",
                "roles.yargi.gpu_layers=-1",
            ]
        )
        config = cli._resolve_config(args)

        self.assertEqual(config["roles"]["yargi"]["backend"], "libmercan")
        self.assertEqual(config["roles"]["yargi"]["model"], "/tmp/yargi.mercan")
        self.assertEqual(config["roles"]["yargi"]["gpu_layers"], -1)

    def test_default_config_has_all_roles(self) -> None:
        self.assertEqual(set(cli.DEFAULT_CONFIG["roles"]), {"yasama", "yurutme", "yargi"})


if __name__ == "__main__":
    unittest.main()
