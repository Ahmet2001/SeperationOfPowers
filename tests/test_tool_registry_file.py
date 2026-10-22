import json
import unittest

from runtime.local_tools import BUILTIN_TOOL_REGISTRY
from runtime.registry_export import REGISTRY_PATH, canonical_registry, render
from yasama.yasama import CANONICAL_ACTIONS


class ToolRegistryFileTests(unittest.TestCase):
    def test_file_matches_the_code(self) -> None:
        self.assertEqual(
            REGISTRY_PATH.read_text(encoding="utf-8"),
            render(),
            "runtime/tool_registry.json is stale; run python -m runtime.registry_export",
        )

    def test_every_canonical_action_is_described_once(self) -> None:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(registry["actions"]), CANONICAL_ACTIONS)
        self.assertEqual(len(registry["actions"]), len(CANONICAL_ACTIONS))
        described = set(registry["tools"]) | set(registry["control_actions"])
        self.assertEqual(described, CANONICAL_ACTIONS)
        self.assertFalse(set(registry["tools"]) & set(registry["control_actions"]))

    def test_tools_carry_public_fields_and_an_executor_kind(self) -> None:
        tools = canonical_registry()["tools"]
        self.assertEqual(set(tools), set(BUILTIN_TOOL_REGISTRY))
        for action, entry in tools.items():
            with self.subTest(action=action):
                self.assertFalse([key for key in entry if key.startswith("_")])
                self.assertIn(entry["executor"], {"local", "web", "external"})
                self.assertEqual(entry["parameters"]["type"], "object")
        self.assertEqual(tools["CALCULATE"]["executor"], "local")
        self.assertEqual(tools["WEB_SEARCH"]["executor"], "web")
        self.assertEqual(tools["SEND_MAIL"]["executor"], "external")


if __name__ == "__main__":
    unittest.main()
