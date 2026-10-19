"""Built-in read-only local tools and default action schemas.

Read-only local actions execute directly for zero-config CLI testing. Actions
with side effects (for example SEND_MAIL or SHELL_EXECUTE) may be present in the
registry so Yurutme can construct arguments, but still require an explicitly
configured external executor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


# Only skip generated dependency/cache trees on broad workspace searches.
SEARCH_IGNORED_DIRS = frozenset({
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", "node_modules",
})


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


# Descriptions are written for the models: what the tool does and when it is
# the right choice.  Whether an executor is configured is reported separately
# (RUNTIME_INFO), so it does not belong in the description text.
BUILTIN_TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "FILE_LIST": {
        "description": (
            "Bir klasördeki dosya ve klasörlerin adlarını listeler; içeriklerini okumaz. "
            "Dosya türü istenmişse deseni istekten çıkar: Python dosyaları için '*.py', "
            "JSON dosyaları için '*.json'. Filtre istenmediyse '*' kullan."
        ),
        "parameters": _schema({
            "path": {"type": "string", "description": "Listelenecek klasör.", "default": "."},
            "pattern": {
                "type": "string",
                "description": "İstekten çıkarılan glob deseni, örneğin '*.py'.",
                "default": "*",
            },
            "recursive": {
                "type": "boolean",
                "description": "Alt klasörler de listelensin mi?",
                "default": False,
            },
        }, []),
        "placeholder": {"path": ".", "pattern": None, "recursive": False},
        "_builtin_executor": "local_readonly",
    },
    "FILE_READ": {
        "description": (
            "Adı bilinen tek bir metin dosyasının içeriğini okur. Hangi dosyada "
            "olduğu bilinmeyen bir metni aramak için FILE_SEARCH kullanılır."
        ),
        "parameters": _schema({
            "path": {"type": "string", "description": "Okunacak dosyanın yolu."},
            "max_chars": {
                "type": "integer",
                "description": "Okunacak en fazla karakter sayısı.",
                "default": 20000,
            },
        }, ["path"]),
        "placeholder": {"path": None, "max_chars": 20000},
        "_builtin_executor": "local_readonly",
    },
    "FILE_SEARCH": {
        "description": (
            "Dosyaların içinde bir kelime veya metin arar ve geçtiği dosya ile "
            "satırları döndürür. Dosya adı bilinip içeriği isteniyorsa FILE_READ kullanılır."
        ),
        "parameters": _schema({
            "query": {"type": "string", "description": "Aranacak kelime veya metin."},
            "path": {"type": "string", "description": "Aramanın yapılacağı klasör.", "default": "."},
            "pattern": {
                "type": "string",
                "description": "Aranacak dosyaları sınırlayan glob deseni.",
                "default": "*",
            },
            "recursive": {
                "type": "boolean",
                "description": "Alt klasörlerde de aransın mı?",
                "default": True,
            },
            "max_results": {
                "type": "integer",
                "description": "En fazla sonuç sayısı.",
                "default": 50,
            },
        }, ["query"]),
        "placeholder": {
            "query": None,
            "path": ".",
            "pattern": "*",
            "recursive": True,
            "max_results": 50,
        },
        "_builtin_executor": "local_readonly",
    },
    "FILE_CREATE": {
        "description": (
            "Yeni bir dosya oluşturur ve içine verilen içeriği yazar. Var olan bir "
            "dosyanın bir kısmını değiştirmek için FILE_EDIT kullanılır."
        ),
        "parameters": _schema({
            "path": {"type": "string", "description": "Oluşturulacak dosyanın yolu."},
            "content": {"type": "string", "description": "Dosyaya yazılacak içerik."},
        }, ["path", "content"]),
        "placeholder": {"path": None, "content": None},
        "_executor_required": True,
    },
    "FILE_EDIT": {
        "description": (
            "Var olan bir dosyada belirli bir metni yenisiyle değiştirir. Yeni dosya "
            "oluşturmak için FILE_CREATE kullanılır."
        ),
        "parameters": _schema({
            "path": {"type": "string", "description": "Düzenlenecek dosyanın yolu."},
            "old_text": {"type": "string", "description": "Dosyada değiştirilecek mevcut metin."},
            "new_text": {"type": "string", "description": "Yerine yazılacak yeni metin."},
        }, ["path", "old_text", "new_text"]),
        "placeholder": {"path": None, "old_text": None, "new_text": None},
        "_executor_required": True,
    },
    "FILE_DELETE": {
        "description": "Bir dosyayı siler. Geri alınamaz; yalnızca kullanıcı açıkça silmeyi istediğinde kullanılır.",
        "parameters": _schema({
            "path": {"type": "string", "description": "Silinecek dosyanın yolu."},
        }, ["path"]),
        "placeholder": {"path": None},
        "_executor_required": True,
    },
    "FILE_MOVE": {
        "description": "Bir dosyayı başka bir klasöre taşır veya adını değiştirir.",
        "parameters": _schema({
            "source": {"type": "string", "description": "Taşınacak dosyanın mevcut yolu."},
            "destination": {"type": "string", "description": "Dosyanın yeni yolu veya hedef klasör."},
        }, ["source", "destination"]),
        "placeholder": {"source": None, "destination": None},
        "_executor_required": True,
    },
    "WEB_SEARCH": {
        "description": (
            "İnternette arama yapar ve sonuç başlıkları ile bağlantıları döndürür. "
            "Adresi belli tek bir sayfayı açmak için WEB_OPEN kullanılır."
        ),
        "parameters": _schema({
            "query": {"type": "string", "description": "Arama sorgusu."},
            "max_results": {"type": "integer", "description": "En fazla sonuç sayısı.", "default": 3},
        }, ["query"]),
        "placeholder": {"query": None, "max_results": 3},
        "_executor_required": True,
    },
    "WEB_OPEN": {
        "description": "Adresi verilen tek bir web sayfasını açıp metnini getirir.",
        "parameters": _schema({
            "url": {"type": "string", "description": "Açılacak sayfanın tam adresi."},
        }, ["url"]),
        "placeholder": {"url": None},
        "_executor_required": True,
    },
    "DOWNLOAD": {
        "description": (
            "Bir adresteki dosyayı indirip diske kaydeder. Sayfanın metnini okumak "
            "için WEB_OPEN kullanılır."
        ),
        "parameters": _schema({
            "url": {"type": "string", "description": "İndirilecek dosyanın adresi."},
            "path": {
                "type": "string",
                "description": "Dosyanın kaydedileceği klasör veya yol.",
                "default": ".",
            },
        }, ["url"]),
        "placeholder": {"url": None, "path": "."},
        "_executor_required": True,
    },
    "SHELL_EXECUTE": {
        "description": (
            "Kullanıcının istediği terminal komutunu çalıştırır. Python kodu "
            "çalıştırmak için CODE_EXECUTE kullanılır."
        ),
        "parameters": _schema({
            "command": {"type": "string", "description": "Kullanıcının istediği komutun tam hali."},
            "cwd": {"type": "string", "description": "Komutun çalışacağı klasör.", "default": "."},
        }, ["command"]),
        "placeholder": {"command": None, "cwd": "."},
        "_executor_required": True,
    },
    "CODE_EXECUTE": {
        "description": (
            "Verilen kod parçasını çalıştırır ve çıktısını döndürür. Terminal "
            "komutları için SHELL_EXECUTE kullanılır."
        ),
        "parameters": _schema({
            "language": {"type": "string", "description": "Programlama dili, örneğin python."},
            "code": {"type": "string", "description": "Çalıştırılacak kod."},
        }, ["language", "code"]),
        "placeholder": {"language": "python", "code": None},
        "_executor_required": True,
    },
    "SEND_MAIL": {
        "description": (
            "E-posta gönderir. Alıcı adresi, konu ve içerik kullanıcının isteğinde "
            "açıkça bulunmalıdır."
        ),
        "parameters": _schema({
            "to": {"type": "string", "description": "Alıcının e-posta adresi."},
            "subject": {"type": "string", "description": "E-postanın konusu."},
            "body": {"type": "string", "description": "Düz metin e-posta içeriği."},
        }, ["to", "subject", "body"]),
        "placeholder": {"to": None, "subject": None, "body": None},
        "_executor_required": True,
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
                data={
                    "path": str(base.relative_to(root)) or ".",
                    "pattern": pattern,
                    "recursive": recursive,
                    "entries": entries,
                },
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
                if any(
                    part in SEARCH_IGNORED_DIRS
                    for part in path.relative_to(base).parts[:-1]
                ):
                    continue
                if not path.is_file():
                    continue
                # A symlinked file can point outside the approved workspace.
                try:
                    path.resolve().relative_to(root)
                except ValueError:
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
