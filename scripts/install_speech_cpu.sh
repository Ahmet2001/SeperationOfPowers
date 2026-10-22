#!/usr/bin/env bash
# Install the optional EMA Lightning TTS using official CPU-only PyTorch wheels.
#
# Usage from any working directory:
#   source /path/to/SeperationOfPowers/.venv/bin/activate
#   bash /path/to/SeperationOfPowers/scripts/install_speech_cpu.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

"$PYTHON" - <<'PY'
import sys

if sys.prefix == sys.base_prefix:
    raise SystemExit(
        "Activate a virtual environment first: source .venv/bin/activate"
    )
if not ((3, 11) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(
        "ema-lightning requires Python 3.11, 3.12 or 3.13. "
        "Create a compatible .venv and rerun this script."
    )
PY

echo "[speech-cpu] Installing CPU-only PyTorch from the official PyTorch wheel index..."
"$PYTHON" -m pip install --upgrade --force-reinstall 'torch>=2.1' \
  --index-url https://download.pytorch.org/whl/cpu

echo "[speech-cpu] Installing EMA Lightning and sounddevice..."
"$PYTHON" -m pip install -e "$PROJECT_ROOT[speech]"

echo "[speech-cpu] Verifying the installed PyTorch build..."
"$PYTHON" - <<'PY'
import torch

cuda = torch.version.cuda
hip = getattr(torch.version, "hip", None)
if cuda is not None or hip is not None:
    raise SystemExit(
        "ERROR: A CUDA/ROCm PyTorch build is still installed. "
        "Use a clean virtual environment and rerun this script."
    )
print(f"[speech-cpu] OK: torch={torch.__version__}, cuda={cuda}, hip={hip}")
print("[speech-cpu] EMA Lightning can now run with device='cpu'.")

# Upstream EMA model weights are not loaded or downloaded during installation.
PY

echo "[speech-cpu] Ready. Run 'mercan-sop chat --speech ...' to enable voice."
