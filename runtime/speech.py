"""Optional Turkish text-to-speech using canberkkkkkk/ema-lightning.

This adapter is intentionally separate from Yasama/Yurutme/Yargi inference.
It voices only the final user-visible answer, never tool arguments or secrets
in debug traces. Dependencies and model weights are loaded on first use, only
when the CLI's --speech mode is enabled.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


MODEL_REPO = "canberkkkkkk/ema-lightning"


def _load_ema() -> Any:
    try:
        import torch
        from ema_lightning import EMA
    except ImportError as exc:
        raise RuntimeError(
            "Speech için EMA Lightning ve CPU-only PyTorch gerekli. "
            "Kurulum: bash scripts/install_speech_cpu.sh "
            "(Python 3.11+ gerektirir)."
        ) from exc

    # Avoid silently accepting a CUDA/ROCm build in the CPU-only speech mode.
    # PyTorch CPU wheel uses torch.version.cuda == torch.version.hip == None.
    if torch.version.cuda is not None or getattr(torch.version, "hip", None) is not None:
        raise RuntimeError(
            "Speech için GPU destekli PyTorch bulundu. "
            "CPU-only PyTorch kurmak için: bash scripts/install_speech_cpu.sh"
        )

    # The upstream EMA constructor downloads model weights from
    # canberkkkkkk/ema-lightning once, then reuses the local HF cache.
    # Explicitly use CPU, regardless of whether a GPU is available.
    return EMA(device="cpu")


def _play_audio(audio: Any, sample_rate: int) -> None:
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Hoparlörden çalmak için sounddevice/PortAudio gerekli. "
            'Kurulum: python -m pip install -e ".[speech]" '
            "(Ubuntu: sudo apt install libportaudio2)."
        ) from exc

    sd.play(audio, samplerate=sample_rate)
    sd.wait()


class EMALightningSpeech:
    """Lazy, reusable EMA TTS instance for chat and one-shot CLI output."""

    def __init__(
        self,
        *,
        ema_factory: Callable[[], Any] | None = None,
        playback: Callable[[Any, int], None] | None = None,
    ) -> None:
        self._ema_factory = ema_factory or _load_ema
        self._playback = playback or _play_audio
        self._ema: Any | None = None

    def speak(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            return

        if self._ema is None:
            self._ema = self._ema_factory()

        speech = self._ema.say(text.strip())
        self._playback(speech.audio, speech.sample_rate)
