"""Physical tool registry: ``runtime/tool_registry.json`` generated from code.

The code is the single source (``BUILTIN_TOOL_REGISTRY`` and Yasama's action
order); the JSON file is what data pipelines and documents read. A test fails
when the file and the code differ, so regenerate it after changing a tool:

    python -m runtime.registry_export          # rewrite the file
    python -m runtime.registry_export --check  # exit 1 if it is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from runtime.local_tools import BUILTIN_TOOL_REGISTRY
from runtime.web_tools import WEB_ACTIONS

REGISTRY_FORMAT = 1
CONTRACT_VERSION = "0.5"  # PROJECT_MASTER.md version the file belongs to
REGISTRY_PATH = Path(__file__).with_name("tool_registry.json")


def _executor(action: str, entry: dict[str, Any]) -> str:
    if entry.get("_builtin_executor") == "local_readonly":
        return "local"
    if action in WEB_ACTIONS:
        return "web"  # built in when a SearXNG URL is configured
    return "external"


def canonical_registry() -> dict[str, Any]:
    from yasama.yasama import ACTION_ORDER, CONTROL_ACTION_DESCRIPTIONS

    tools = {
        action: {
            **{key: value for key, value in BUILTIN_TOOL_REGISTRY[action].items()
               if not key.startswith("_")},
            "executor": _executor(action, BUILTIN_TOOL_REGISTRY[action]),
        }
        for action in ACTION_ORDER
        if action not in CONTROL_ACTION_DESCRIPTIONS
    }
    return {
        "format": REGISTRY_FORMAT,
        "contract": CONTRACT_VERSION,
        "actions": list(ACTION_ORDER),
        "control_actions": dict(CONTROL_ACTION_DESCRIPTIONS),
        "tools": tools,
    }


def render() -> str:
    return json.dumps(canonical_registry(), ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    current = REGISTRY_PATH.read_text(encoding="utf-8") if REGISTRY_PATH.exists() else None
    if "--check" in args:
        if current != render():
            print(f"{REGISTRY_PATH} is stale; run python -m runtime.registry_export")
            return 1
        return 0
    REGISTRY_PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {REGISTRY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
