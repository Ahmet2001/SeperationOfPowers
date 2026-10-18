"""In-process Mercan adapter using EthosoftLib's public, stateless API.

EthosoftLib owns native bindings, backend lifetime, formatting and sampling.
Install the optional ``mercan`` extra or an editable EthosoftLib checkout.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .backends import GenerationConfig


class LibMercanBackend:
    """ChatBackend over ethosoftlib.mercan and a local .mercan model.

    Every generate() receives the complete transcript. SDK chat history and
    default system prompts are never inserted into role requests.
    """

    def __init__(
        self, model: str | os.PathLike[str], *,
        library: str | os.PathLike[str] | None = None,
        device: str = "auto",
        gpu_layers: int | None = None, use_mmap: bool | None = None,
        check_tensors: bool | None = None, n_ctx: int | None = None,
        n_batch: int | None = None, threads: int | None = None,
        threads_batch: int | None = None,
        plugins: Sequence[str | os.PathLike[str]] | None = None,
        role_system: str = "sistem", role_user: str = "kullanici",
        role_assistant: str = "asistan",
    ) -> None:
        self.model_path = str(Path(model).expanduser())
        self._sdk_model = None
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"libmercan requires a local model file; not found: {self.model_path}")
        try:
            from ethosoftlib.mercan import MercanRuntime, Model
        except ImportError as exc:
            raise RuntimeError(
                "libmercan requires EthosoftLib. Install with pip install -e '.[mercan]' "
                "or pip install -e /path/to/EthosoftLib."
            ) from exc
        if (getattr(Model, "NATIVE_GENERATION_API_VERSION", 0) != 1
                or getattr(Model, "DEVICE_API_VERSION", 0) != 1):
            raise RuntimeError("EthosoftLib >=0.7.0 with native generation/device API v1 is required; update the SDK.")
        runtime = MercanRuntime(library_path=library)
        native_model = None
        try:
            runtime.require_generation_api()
            for plugin in plugins or ():
                runtime.load_plugin(plugin)
            native_model = runtime.load_model(
                self.model_path, device=device, gpu_layers=gpu_layers,
                use_mmap=use_mmap, check_tensors=check_tensors,
            )
            self._sdk_model = Model(
                runtime, native_model, Path(self.model_path), system_prompt="",
                n_ctx=n_ctx, n_batch=n_batch, threads=threads,
                threads_batch=threads_batch, role_system=role_system,
                role_user=role_user, role_assistant=role_assistant,
            )
        except BaseException:
            if native_model is not None:
                native_model.close()
            runtime.close()
            raise

    @property
    def device(self) -> str:
        if self._sdk_model is None:
            raise RuntimeError("LibMercanBackend is closed.")
        return str(self._sdk_model.info()["device"])

    @property
    def version(self) -> str:
        if self._sdk_model is None:
            raise RuntimeError("LibMercanBackend is closed.")
        return str(self._sdk_model.info()["runtime_version"])

    def generate(
        self, messages: Sequence[Mapping[str, str]], *,
        config: GenerationConfig | None = None,
    ) -> str:
        if self._sdk_model is None:
            raise RuntimeError("LibMercanBackend is closed.")
        cfg = config or GenerationConfig()
        text = self._sdk_model.generate(
            messages, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
            top_k=cfg.top_k, top_p=cfg.top_p,
            repeat_penalty=cfg.repeat_penalty, repeat_last_n=cfg.repeat_last_n,
        )
        # Checkpoints whose vocab lacks the ChatML special tokens emit
        # "<|im_end|>" as plain text, which the native stop check cannot see.
        return text.split("<|im_end|>", 1)[0]

    def close(self) -> None:
        model = self._sdk_model
        if model is not None:
            model.close()
            self._sdk_model = None

    def __enter__(self) -> LibMercanBackend:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
