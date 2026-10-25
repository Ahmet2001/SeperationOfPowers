"""Built-in read-only local tools and default action schemas.

Read-only local actions execute directly for zero-config CLI testing. Actions
with side effects (for example SEND_MAIL or SHELL_EXECUTE) may be present in the
registry so Yurutme can construct arguments, but still require an explicitly
configured external executor.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

# ethosoftlib (rag; math needs sympy) is imported where it is used: the
# pipeline itself has no dependencies, and only long FILE_READs and CALCULATE
# need the library (install the "mercan" extra).


# Only skip generated dependency/cache trees on broad workspace searches.
SEARCH_IGNORED_DIRS = frozenset({
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", "node_modules",
})

# A file up to this many characters is returned whole; a longer one (and every
# PDF past it) is answered with its best-matching chunks up to the same total.
# Serialized observations measured 2.1-2.7 characters per Mercan token, so 7000
# characters stay under ~3.4k tokens: with Yargi's ~2.4k prompt and 768 answer
# tokens (Yurutme: ~2.9k + 512) that leaves room in the 8k context for a second
# small observation.
READ_CHAR_BUDGET = 7000

# Integers from here on are reported in scientific notation: a 200-digit
# result was copied digit by digit into the answer.
MAX_EXACT_INTEGER = 10**21


@lru_cache(maxsize=1)
def _request_terms() -> frozenset[str]:
    """Request words that say what to do with the file, not what is in it, as
    ethosoftlib.rag search terms (five-letter prefixes). With them a question
    like "X.md dosyasında ... ne diyor?" ranked the file's title chunk first."""
    from ethosoftlib.rag import tokenize

    return frozenset(tokenize(
        "dosya dosyası dosyasında dosyasını oku okuyup göster anlat açıkla söyle "
        "özetle diyor yazıyor nedir neler hangi göre içinde içeriği bul"
    ))


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
            "Adı bilinen tek bir dosyayı (metin veya PDF) okur. Kısa dosyanın tamamını, "
            "uzun dosyanın soruyla en ilgili bölümlerini satır numaralarıyla getirir. "
            "Hangi dosyada olduğu bilinmeyen bir metni aramak için FILE_SEARCH kullanılır."
        ),
        "parameters": _schema({
            "path": {"type": "string", "description": "Okunacak dosyanın yolu."},
            "query": {
                "type": "string",
                "description": (
                    "Uzun dosyada aranacak konu. Boş bırakılırsa kullanıcının isteği kullanılır."
                ),
            },
        }, ["path"]),
        "placeholder": {"path": None},
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
    # Schema of the training data's largest calculator source (tool-math
    # `calc {expression}`); its operation/a/b rows convert to expressions
    # without loss (PROJECT_MASTER §6).
    "CALCULATE": {
        "description": (
            "Bir matematik ifadesini hesaplar ve sonucu döndürür: toplama, çıkarma, "
            "çarpma, bölme, üs alma, karekök. Sayıları kullanıcının isteğinden al; "
            "yazıyla yazılmış sayıları rakama çevir."
        ),
        "parameters": _schema({
            "expression": {
                "type": "string",
                "description": (
                    "Rakamlar, + - * / ^ ve parantezlerle yazılmış ifade; karekök için "
                    "sqrt(). Örnek: '45.5*12', '(300-87)/3', 'sqrt(625)', '2^16'."
                ),
            },
        }, ["expression"]),
        "placeholder": {"expression": None},
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


def _file_sections(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Text of ``path`` as (text, metadata) parts: one per PDF page, else one."""
    if path.suffix.lower() == ".pdf":
        from ethosoftlib import ocr

        document = ocr.read(path, allow_missing=True)
        return [(page.text, {"page": page.number}) for page in document.pages if page.text]
    with path.open("rb") as handle:
        if b"\0" in handle.read(4096):
            raise ValueError("Bu bir metin dosyası değil; okunamıyor.")
    return [(path.read_text(encoding="utf-8", errors="replace"), {})]


def _retrieval_query(query: str, user_prompt: str, relative: str) -> str:
    """Search terms for a long file: Yurutme's query plus the user's request
    (Yurutme queries were seen translated to English or made up), without
    the file's own name and request words."""
    text = f"{query} {user_prompt}"
    # Only the written path goes: "kurulumu" stays in a question about kurulum.md.
    for name in sorted({relative, Path(relative).name}, key=len, reverse=True):
        text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)
    from ethosoftlib.rag import tokenize

    terms: list[str] = []
    request_terms = _request_terms()
    for term in tokenize(text):
        if term not in request_terms and term not in terms:
            terms.append(term)
    return " ".join(terms)


def _read_file(path: Path, relative: str, query: str, user_prompt: str) -> dict[str, Any]:
    action = "FILE_READ"
    sections = _file_sections(path)
    total_chars = sum(len(text) for text, _ in sections)
    if total_chars <= READ_CHAR_BUDGET:
        return _observation(action, data={
            "path": relative,
            "content": "\n\n".join(text for text, _ in sections),
        })

    from ethosoftlib.rag import Hit, Index

    query = _retrieval_query(query, user_prompt, relative)
    index = Index()
    for text, metadata in sections:
        index.add(text, relative, metadata=metadata)
    hits = index.search(query, k=len(index)) if query else []
    matched = bool(hits)
    if not matched:
        # No query term occurs in the file: show its beginning instead.
        hits = [Hit(chunk.text, chunk.source, chunk.line, 0.0, dict(chunk.metadata))
                for chunk in index.chunks]
    picked: list[Hit] = []
    used = 0
    for hit in hits:
        if used + len(hit.text) > READ_CHAR_BUDGET:
            if picked:
                break
            continue
        picked.append(hit)
        used += len(hit.text)
    picked.sort(key=lambda hit: (hit.metadata.get("page", 0), hit.line))
    # One content string, as in the training data's FILE_READ observations:
    # given a list of section objects, Yargi dumped them back as JSON.
    content = "\n\n".join(
        (f"[sayfa {hit.metadata['page']}, satır {hit.line}]" if "page" in hit.metadata
         else f"[satır {hit.line}]") + "\n" + hit.text
        for hit in picked
    )
    return _observation(action, data={
        "path": relative,
        "content": content,
        "partial": True,
        "query": query,
        "matched": matched,
        "total_chars": total_chars,
    })


def _calculate(arguments: Mapping[str, Any]) -> dict[str, Any]:
    action = "CALCULATE"
    expression = str(arguments.get("expression") or "").strip()
    if not expression:
        return _observation(action, error="Hesaplanacak ifade boş.")
    try:
        from ethosoftlib.math import evaluate
        from ethosoftlib.math.protocol import ToolError
    except ImportError as exc:
        return _observation(action, error=(
            "CALCULATE için ethosoftlib[math] (sympy) kurulu olmalıdır: " + str(exc)
        ))
    # evaluate() parses the expression itself (no eval); exact results come
    # back as Fraction, irrational ones (sqrt(2)) as sympy numbers.
    try:
        value = evaluate(expression)
    except ToolError as exc:
        return _observation(action, error=f"İfade hesaplanamadı: {exc}")
    if isinstance(value, bool):
        return _observation(action, data={"result": value})
    try:
        result = _plain_number(value)
    except OverflowError:
        return _observation(action, error="Sonuç çok büyük; hesaplanamadı.")
    except (TypeError, ValueError):
        return _observation(action, error=(
            f"İfade bir sayıya hesaplanamadı: {value}. Bilinmeyen içermeyen bir ifade yaz."
        ))
    return _observation(action, data={"result": result})


def _plain_number(value: Any) -> int | float:
    """JSON number for an evaluate() result: integers exact up to
    MAX_EXACT_INTEGER, everything else 12 significant digits, which hides
    binary noise (0.1 + 0.2) without changing a result a person writes down."""
    if isinstance(value, Fraction):
        if value.denominator == 1 and abs(value.numerator) < MAX_EXACT_INTEGER:
            return value.numerator
        number = float(value)
    else:
        if getattr(value, "free_symbols", None) or not getattr(value, "is_real", False):
            raise ValueError(value)
        number = float(value.evalf(17))
    if not math.isfinite(number):
        raise OverflowError(value)
    number = float(f"{number:.12g}")
    if number.is_integer() and abs(number) < MAX_EXACT_INTEGER:
        return int(number)
    return number


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
    user_prompt: str | None = None,
) -> dict[str, Any]:
    """Execute one built-in read-only action inside the current workspace.

    ``user_prompt`` is the retrieval query of a long FILE_READ when Yurutme
    left ``query`` empty, as the training data's FILE_READ calls do.
    """

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
            relative = str(path.relative_to(root))
            return _read_file(
                path, relative, str(arguments.get("query") or ""), user_prompt or ""
            )

        if action == "CALCULATE":
            return _calculate(arguments)

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
