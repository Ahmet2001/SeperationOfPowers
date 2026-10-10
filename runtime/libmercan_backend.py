"""Direct in-process backend for the custom libmercan C runtime.

This adapter mirrors the generation loop used by ``mercanApp-test1/cli/main.cpp``:
model load -> tokenize -> decode prompt -> read logits -> sample -> decode token.

It deliberately targets the public C ABI declared in ``runtime/libmercan/include/mercan.h``
so Yasama/Yurutme/Yargi do not depend on the Mercan CLI process boundary.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import heapq
import math
import os
import random
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .backends import GenerationConfig


class _MercanModelParams(ctypes.Structure):
    _fields_ = [
        ("n_gpu_layers", ctypes.c_int32),
        ("use_mmap", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
    ]


class _MercanContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
    ]


class LibMercanBackend:
    """ChatBackend implementation backed directly by ``libmercan`` via ctypes.

    Parameters
    ----------
    model:
        Local ``.mercan`` model path. Unlike the Mercan CLI, this direct adapter does
        not perform Hugging Face downloads before calling ``mercan_model_load``.
    library:
        Path/name of the shared library. If omitted, ``find_library('mercan')`` and
        common platform filenames are tried.
    plugins:
        Optional architecture/tokenizer plugins loaded through
        ``mercan_plugin_load_v1`` before the model is opened.

    Notes
    -----
    ``mercan_backend_init`` is process-global in the underlying runtime. We initialize
    each loaded shared library once and intentionally leave the global backend alive
    for process lifetime; individual model/context handles are still freed normally.
    """

    _global_lock = threading.RLock()
    _initialized_libraries: set[str] = set()

    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        library: str | os.PathLike[str] | None = None,
        gpu_layers: int | None = None,
        use_mmap: bool | None = None,
        check_tensors: bool | None = None,
        n_ctx: int | None = None,
        n_batch: int | None = None,
        threads: int | None = None,
        threads_batch: int | None = None,
        plugins: Sequence[str | os.PathLike[str]] | None = None,
        role_system: str = "sistem",
        role_user: str = "kullanici",
        role_assistant: str = "asistan",
    ) -> None:
        self.model_path = str(Path(model).expanduser())
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(
                f"libmercan requires a local model file; not found: {self.model_path}"
            )

        self.library_path = self._resolve_library(library)
        self.role_system = role_system
        self.role_user = role_user
        self.role_assistant = role_assistant
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.threads = threads
        self.threads_batch = threads_batch
        self._lock = threading.RLock()
        self._closed = False

        try:
            self._lib = ctypes.CDLL(self.library_path)
        except OSError as exc:
            raise RuntimeError(
                f"Could not load libmercan shared library {self.library_path!r}: {exc}"
            ) from exc

        self._configure_api(self._lib)
        self._initialize_backend_once()
        self._load_plugins(plugins or [])

        params = self._lib.mercan_model_default_params()
        if gpu_layers is not None:
            params.n_gpu_layers = gpu_layers
        if use_mmap is not None:
            params.use_mmap = use_mmap
        if check_tensors is not None:
            params.check_tensors = check_tensors

        self._model = self._lib.mercan_model_load(
            os.fsencode(self.model_path),
            params,
        )
        if not self._model:
            raise RuntimeError(f"libmercan model load failed: {self._last_error()}")

    @staticmethod
    def _resolve_library(library: str | os.PathLike[str] | None) -> str:
        if library is not None:
            return os.fspath(library)

        discovered = ctypes.util.find_library("mercan")
        if discovered:
            return discovered

        candidates = (
            "libmercan.so",
            "libmercan.dylib",
            "mercan.dll",
            "libmercan.dll",
        )
        for candidate in candidates:
            if Path(candidate).is_file():
                return str(Path(candidate).resolve())

        raise RuntimeError(
            "libmercan shared library was not found. Pass library='/path/to/libmercan.so' "
            "(or .dylib/.dll) and build mercanApp-test1 with MERCAN_BUILD_SHARED=ON."
        )

    @staticmethod
    def _configure_api(lib: ctypes.CDLL) -> None:
        lib.mercan_version.argtypes = []
        lib.mercan_version.restype = ctypes.c_char_p
        lib.mercan_last_error.argtypes = []
        lib.mercan_last_error.restype = ctypes.c_char_p

        lib.mercan_backend_init.argtypes = []
        lib.mercan_backend_init.restype = None
        lib.mercan_backend_free.argtypes = []
        lib.mercan_backend_free.restype = None

        lib.mercan_model_default_params.argtypes = []
        lib.mercan_model_default_params.restype = _MercanModelParams
        lib.mercan_context_default_params.argtypes = []
        lib.mercan_context_default_params.restype = _MercanContextParams

        lib.mercan_model_load.argtypes = [ctypes.c_char_p, _MercanModelParams]
        lib.mercan_model_load.restype = ctypes.c_void_p
        lib.mercan_model_free.argtypes = [ctypes.c_void_p]
        lib.mercan_model_free.restype = None

        lib.mercan_context_create.argtypes = [ctypes.c_void_p, _MercanContextParams]
        lib.mercan_context_create.restype = ctypes.c_void_p
        lib.mercan_context_free.argtypes = [ctypes.c_void_p]
        lib.mercan_context_free.restype = None

        lib.mercan_tokenize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
        ]
        lib.mercan_tokenize.restype = ctypes.c_int32
        lib.mercan_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
        ]
        lib.mercan_decode.restype = ctypes.c_int32
        lib.mercan_logits.argtypes = [ctypes.c_void_p]
        lib.mercan_logits.restype = ctypes.POINTER(ctypes.c_float)
        lib.mercan_vocab_size.argtypes = [ctypes.c_void_p]
        lib.mercan_vocab_size.restype = ctypes.c_int32
        lib.mercan_context_size.argtypes = [ctypes.c_void_p]
        lib.mercan_context_size.restype = ctypes.c_uint32
        lib.mercan_eos_token.argtypes = [ctypes.c_void_p]
        lib.mercan_eos_token.restype = ctypes.c_int32
        lib.mercan_token_to_piece.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int32,
            ctypes.c_bool,
        ]
        lib.mercan_token_to_piece.restype = ctypes.c_int32

        # Plugin symbols are part of mercan_plugin.h and may be absent in older builds.
        if hasattr(lib, "mercan_plugin_load_v1"):
            lib.mercan_plugin_load_v1.argtypes = [ctypes.c_char_p]
            lib.mercan_plugin_load_v1.restype = ctypes.c_int
            lib.mercan_plugin_last_error_v1.argtypes = []
            lib.mercan_plugin_last_error_v1.restype = ctypes.c_char_p

    def _initialize_backend_once(self) -> None:
        key = os.path.realpath(self.library_path)
        with self._global_lock:
            if key not in self._initialized_libraries:
                self._lib.mercan_backend_init()
                self._initialized_libraries.add(key)

    def _load_plugins(self, plugins: Sequence[str | os.PathLike[str]]) -> None:
        if not plugins:
            return
        if not hasattr(self._lib, "mercan_plugin_load_v1"):
            raise RuntimeError(
                "This libmercan build does not expose mercan_plugin_load_v1."
            )
        for plugin in plugins:
            rc = self._lib.mercan_plugin_load_v1(os.fsencode(os.fspath(plugin)))
            if rc < 0:
                raw = self._lib.mercan_plugin_last_error_v1()
                detail = raw.decode("utf-8", errors="replace") if raw else "unknown error"
                raise RuntimeError(f"libmercan plugin load failed: {detail}")

    @property
    def version(self) -> str:
        raw = self._lib.mercan_version()
        return raw.decode("utf-8", errors="replace") if raw else "unknown"

    def close(self) -> None:
        with self._lock:
            if not self._closed and getattr(self, "_model", None):
                self._lib.mercan_model_free(self._model)
                self._model = None
                self._closed = True

    def __enter__(self) -> "LibMercanBackend":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        cfg = config or GenerationConfig()
        formatted = self._format_chat(
            messages,
            role_system=self.role_system,
            role_user=self.role_user,
            role_assistant=self.role_assistant,
        )

        with self._lock:
            if self._closed or not self._model:
                raise RuntimeError("LibMercanBackend is closed.")

            cp = self._lib.mercan_context_default_params()
            if self.n_ctx is not None:
                cp.n_ctx = self.n_ctx
            if self.n_batch is not None:
                cp.n_batch = self.n_batch
            if self.threads is not None:
                cp.n_threads = self.threads
            if self.threads_batch is not None:
                cp.n_threads_batch = self.threads_batch
            elif self.threads is not None:
                cp.n_threads_batch = self.threads

            ctx = self._lib.mercan_context_create(self._model, cp)
            if not ctx:
                raise RuntimeError(
                    f"libmercan context creation failed: {self._last_error()}"
                )

            try:
                return self._generate_with_context(ctx, formatted, cfg, cp)
            finally:
                self._lib.mercan_context_free(ctx)

    def _generate_with_context(
        self,
        ctx: ctypes.c_void_p,
        formatted_prompt: str,
        cfg: GenerationConfig,
        cp: _MercanContextParams,
    ) -> str:
        prompt_tokens = self._tokenize(formatted_prompt)
        if not prompt_tokens:
            raise RuntimeError("libmercan prompt produced no tokens.")

        context_size = int(self._lib.mercan_context_size(ctx))
        if context_size > 0 and len(prompt_tokens) >= context_size:
            raise RuntimeError(
                f"Prompt uses {len(prompt_tokens)} tokens but context size is {context_size}."
            )

        batch_size = max(1, int(cp.n_batch))
        for start in range(0, len(prompt_tokens), batch_size):
            chunk = prompt_tokens[start : start + batch_size]
            token_array = (ctypes.c_int32 * len(chunk))(*chunk)
            if self._lib.mercan_decode(ctx, token_array, len(chunk)) != 0:
                raise RuntimeError(
                    f"libmercan prompt decode failed: {self._last_error()}"
                )

        n_vocab = int(self._lib.mercan_vocab_size(self._model))
        if n_vocab <= 0:
            raise RuntimeError("libmercan returned an invalid vocabulary size.")

        eos = int(self._lib.mercan_eos_token(self._model))
        message_start = self._special_token_id("<|im_start|>")
        message_end = self._special_token_id("<|im_end|>")
        stop_tokens = {token for token in (eos, message_start, message_end) if token >= 0}

        history_cap = max(0, int(cfg.repeat_last_n))
        recent_tokens = prompt_tokens[-history_cap:] if history_cap else []
        rng = random.Random()
        pieces: list[str] = []

        remaining_context = (
            max(0, context_size - len(prompt_tokens)) if context_size > 0 else cfg.max_tokens
        )
        max_tokens = min(max(0, int(cfg.max_tokens)), remaining_context)
        if max_tokens <= 0:
            raise RuntimeError("No context capacity remains for generated tokens.")

        for _ in range(max_tokens):
            logits_ptr = self._lib.mercan_logits(ctx)
            if not logits_ptr:
                raise RuntimeError("libmercan returned no logits.")

            next_token = self._sample_token(
                logits_ptr=logits_ptr,
                n_vocab=n_vocab,
                temperature=float(cfg.temperature),
                top_k=int(cfg.top_k),
                top_p=float(cfg.top_p),
                repeat_penalty=float(cfg.repeat_penalty),
                recent_tokens=recent_tokens,
                rng=rng,
            )
            if next_token in stop_tokens:
                break

            pieces.append(self._token_piece(next_token))

            if history_cap:
                recent_tokens.append(next_token)
                if len(recent_tokens) > history_cap:
                    del recent_tokens[0]

            token_array = (ctypes.c_int32 * 1)(next_token)
            if self._lib.mercan_decode(ctx, token_array, 1) != 0:
                raise RuntimeError(
                    f"libmercan generation decode failed: {self._last_error()}"
                )

        return "".join(pieces)

    def _tokenize(self, text: str) -> list[int]:
        encoded = text.encode("utf-8")
        probe = self._lib.mercan_tokenize(
            self._model,
            encoded,
            len(encoded),
            False,
            True,
            None,
            0,
        )
        if probe == 0 and text:
            raise RuntimeError(f"libmercan tokenization failed: {self._last_error()}")

        required = abs(int(probe))
        capacity = max(1, required)
        tokens = (ctypes.c_int32 * capacity)()
        got = self._lib.mercan_tokenize(
            self._model,
            encoded,
            len(encoded),
            False,
            True,
            tokens,
            capacity,
        )
        if got < 0:
            capacity = -int(got)
            tokens = (ctypes.c_int32 * capacity)()
            got = self._lib.mercan_tokenize(
                self._model,
                encoded,
                len(encoded),
                False,
                True,
                tokens,
                capacity,
            )
        if got < 0:
            raise RuntimeError("libmercan token buffer sizing failed.")
        return [int(tokens[index]) for index in range(int(got))]

    def _special_token_id(self, text: str) -> int:
        tokens = self._tokenize(text)
        return tokens[0] if len(tokens) == 1 else -1

    def _token_piece(self, token: int) -> str:
        small = ctypes.create_string_buffer(64)
        count = self._lib.mercan_token_to_piece(
            self._model,
            token,
            small,
            len(small),
            False,
        )
        if count < 0:
            buffer = ctypes.create_string_buffer(-int(count))
            count = self._lib.mercan_token_to_piece(
                self._model,
                token,
                buffer,
                len(buffer),
                False,
            )
            if count < 0:
                raise RuntimeError("libmercan token_to_piece buffer sizing failed.")
            raw = buffer.raw[: int(count)]
        else:
            raw = small.raw[: int(count)]
        return raw.decode("utf-8", errors="replace")

    def _last_error(self) -> str:
        raw = self._lib.mercan_last_error()
        return raw.decode("utf-8", errors="replace") if raw else "unknown error"

    @staticmethod
    def _sample_token(
        *,
        logits_ptr: ctypes.POINTER(ctypes.c_float),
        n_vocab: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repeat_penalty: float,
        recent_tokens: Sequence[int],
        rng: random.Random,
    ) -> int:
        scores = [float(logits_ptr[index]) for index in range(n_vocab)]

        if repeat_penalty != 1.0:
            for token in recent_tokens:
                if 0 <= token < n_vocab:
                    value = scores[token]
                    scores[token] = (
                        value / repeat_penalty if value > 0.0 else value * repeat_penalty
                    )

        if temperature <= 0.0:
            return max(range(n_vocab), key=scores.__getitem__)

        k = max(1, min(top_k, n_vocab))
        candidate_ids = heapq.nlargest(k, range(n_vocab), key=scores.__getitem__)
        max_logit = scores[candidate_ids[0]]
        weights = [
            math.exp((scores[token] - max_logit) / temperature)
            for token in candidate_ids
        ]

        total = sum(weights)
        if 0.0 < top_p < 1.0 and total > 0.0:
            cumulative = 0.0
            keep = 0
            for weight in weights:
                cumulative += weight / total
                keep += 1
                if cumulative >= top_p:
                    break
            candidate_ids = candidate_ids[:keep]
            weights = weights[:keep]

        if not any(weight > 0.0 for weight in weights):
            return candidate_ids[0]
        return int(rng.choices(candidate_ids, weights=weights, k=1)[0])

    @staticmethod
    def _format_chat(
        messages: Sequence[Mapping[str, str]],
        *,
        role_system: str = "sistem",
        role_user: str = "kullanici",
        role_assistant: str = "asistan",
    ) -> str:
        if not messages:
            raise ValueError("At least one chat message is required.")

        role_map = {
            "system": role_system,
            "user": role_user,
            "assistant": role_assistant,
        }
        chunks: list[str] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in role_map:
                raise ValueError(f"Unsupported chat role: {role!r}")
            if not isinstance(content, str):
                raise TypeError("Chat message content must be a string.")
            chunks.append(
                f"<|im_start|>{role_map[role]}\n{content}<|im_end|>\n"
            )

        chunks.append(f"<|im_start|>{role_assistant}\n")
        return "".join(chunks)
