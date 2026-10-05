"""Built-in read-only local tools for zero-config CLI testing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


BUILTIN_TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "FILE_LIST": {
        "description": "List files in a local directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "pattern": {"type": "string", "default": "*"},
                "recursive": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        },
        "placeholder": {"path": ".", "pattern": "*", "recursive": False},
        "_builtin_executor": "local_readonly",
    },
    "FILE_READ": {
        "description": "Read a UTF-8 text file from the local workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer", "default": 20000},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "placeholder": {"path": None, "max_chars": 20000},
        "_builtin_executor": "local_readonly",
    },
    "FILE_SEARCH": {
        "description": "Search text inside files in the local workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "pattern": {"type": "string", "default": "*"},
                "recursive": {"type": "boolean", "default": True},
                "max_results": {"type": "integer", "default": 50},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "placeholder": {
            "query": None,
            "path": ".",
            "pattern": "*",
            "recursive": True,
            "max_results": 50,
        },
        "_builtin_executor": "local_readonly",
    },
}


def _observation(action: str, *, data: Mapping[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
    return {
        "action": action,
        "data": dict(data or {}),
        "status": "error" if error else "success",
        "error": error,
    }


def _resolve_workspace_path(raw_path: Any, workspace: Path) -> Path:
    value = "." if raw_path in {None, ""} else str(raw_path)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Path is outside the workspace: {value}") from exc
    return resolved


def execute_builtin_local(
    action: str,
    arguments: Mapping[str, Any],
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Execute one built-in read-only action inside the current workspace."""

    root = Path(workspace or Path.cwd()).resolve()
    try:
        if action == "FILE_LIST":
            base = _resolve_workspace_path(arguments.get("path", "."), root)
            if not base.is_dir():
                return _observation(action, error=f"Directory not found: {base}")
            pattern = str(arguments.get("pattern") or "*")
            recursive = bool(arguments.get("recursive", False))
            iterator = base.rglob(pattern) if recursive else base.glob(pattern)
            entries = sorted(
                str(path.relative_to(root)) + ("/" if path.is_dir() else "")
                for path in iterator
            )
            return _observation(
                action,
                data={"path": str(base.relative_to(root)) or ".", "entries": entries},
            )

        if action == "FILE_READ":
            path = _resolve_workspace_path(arguments.get("path"), root)
            if not path.is_file():
                return _observation(action, error=f"File not found: {path}")
            max_chars = max(1, min(int(arguments.get("max_chars", 20000)), 200000))
            content = path.read_text(encoding="utf-8", errors="replace")
            truncated = len(content) > max_chars
            return _observation(
                action,
                data={
                    "path": str(path.relative_to(root)),
                    "content": content[:max_chars],
                    "truncated": truncated,
                },
            )

        if action == "FILE_SEARCH":
            query = str(arguments.get("query") or "")
            if not query:
                return _observation(action, error="FILE_SEARCH requires a non-empty query.")
            base = _resolve_workspace_path(arguments.get("path", "."), root)
            if not base.is_dir():
                return _observation(action, error=f"Directory not found: {base}")
            pattern = str(arguments.get("pattern") or "*")
            recursive = bool(arguments.get("recursive", True))
            max_results = max(1, min(int(arguments.get("max_results", 50)), 500))
            iterator = base.rglob(pattern) if recursive else base.glob(pattern)
            matches: list[dict[str, Any]] = []
            for path in iterator:
                if not path.is_file():
                    continue
                try:
                    if path.stat().st_size > 2_000_000:
                        continue
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for line_no, line in enumerate(text.splitlines(), start=1):
                    if query.casefold() in line.casefold():
                        matches.append(
                            {
                                "path": str(path.relative_to(root)),
                                "line": line_no,
                                "text": line[:500],
                            }
                        )
                        if len(matches) >= max_results:
                            return _observation(action, data={"query": query, "matches": matches})
            return _observation(action, data={"query": query, "matches": matches})

        return _observation(action, error=f"No built-in executor for action {action!r}.")
    except Exception as exc:
        return _observation(action, error=str(exc))
