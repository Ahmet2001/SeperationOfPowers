#!/bin/bash
# Export one role checkpoint to .mercan (F16 and Q4_K_M) with the converter
# and llama-quantize the Mercan DPO release used (jobs/export_dpo7895_mercan_q4_hf.slurm).
# CPU only (submit from arf-ui1); nothing is uploaded:
#   sbatch --export=ALL,CKPT=<run>/checkpoints/epoch_02_step_002532.pt,OUT=<dir>,NAME=Mercan-0.8B-Yasama training/export_mercan.slurm
#SBATCH -J mercan_export
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
: "${CKPT:?}" "${OUT:?}" "${NAME:?}"
ROOT=/arf/scratch/egitimg16
NEDO=$ROOT/nedolm_0p8b
IMG=/arf/sw/containers/truba-ai/pytorch-26.06-py3.sif
WORK=$NEDO/jobs/mercan_final_work
QUANTIZE=$ROOT/mercanApp-ciwork/build/llama-tools-final/bin/llama-quantize
F16=$OUT/$NAME-F16.mercan
Q4=$OUT/$NAME-Q4_K_M.mercan
mkdir -p "$OUT"

apptainer exec --bind /arf:/arf "$IMG" python -u "$WORK/mercan_convert_32002.py" "$CKPT" \
  --output "$F16" \
  --vocab-bin "$ROOT/NedoTokenizer/assets/surface-vocab.bin" \
  --vocab-txt "$ROOT/NedoTokenizer/assets/surface-vocab.txt" \
  --role-table "$NEDO/assets/token_role_table_32002_im.pt" \
  --arch-json "$WORK/arch_32002.json" \
  --model-name "$NAME"
apptainer exec --bind /arf:/arf "$IMG" "$QUANTIZE" "$F16" "$Q4" Q4_K_M

for file in "$F16" "$Q4"; do
  python3 - "$file" "$CKPT" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
path, ckpt = Path(sys.argv[1]), Path(sys.argv[2])
with path.open("rb") as f:
    assert f.read(4) == b"GGUF"
h = hashlib.sha256()
with path.open("rb") as f:
    for chunk in iter(lambda: f.read(8 << 20), b""):
        h.update(chunk)
manifest = {
    "format": "mercan", "format_version": 1, "physical_container": "GGUF-v3",
    "quantization": "Q4_K_M" if "Q4_K_M" in path.name else "F16",
    "size_bytes": path.stat().st_size, "sha256": h.hexdigest(), "source_checkpoint": str(ckpt),
    "tokenizer": "NDSRF004",
    "tokenizer_sha256": "72412d981dac65a29d1767bc98821fc2bcffc2de53c534e7c719598515bfb600",
    "vocab_size": 32002, "control_ids": {"message_start": 32000, "message_end": 32001},
    "canonical_roles": ["sistem", "kullanici", "asistan"], "created_unix": int(time.time()),
}
Path(f"{path}.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("EXPORTED", path.name, manifest["size_bytes"], manifest["sha256"])
PY
done
echo EXPORT_OK
