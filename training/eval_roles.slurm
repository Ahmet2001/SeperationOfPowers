#!/bin/bash
# Generation eval of the three role checkpoints and of the DPO base as Yargi,
# one GPU, sequentially (submit from cuda-ui):
#   sbatch --export=ALL,CODE=...,MIX=...,YASAMA=...,YURUTME=...,YARGI=...,BASE=...,OUT=... training/eval_roles.slurm
#SBATCH -J mercan_eval
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg16
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -C H200
#SBATCH --time=06:00:00
set -euo pipefail
: "${CODE:?}" "${MIX:?}" "${YASAMA:?}" "${YURUTME:?}" "${YARGI:?}" "${BASE:?}" "${OUT:?}" "${ETHOSOFTLIB:?}"
IMG=/arf/sw/containers/truba-ai/pytorch-26.06-py3.sif
CACHE=/tmp/$USER/mercan-eval-$SLURM_JOB_ID
mkdir -p "$OUT" "$CACHE"
export TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor CUDA_CACHE_PATH=$CACHE/nv
export PYTHONUNBUFFERED=1
run() {
  apptainer exec --nv --bind /arf:/arf --bind /tmp:/tmp --env PYTHONPATH="$CODE:$ETHOSOFTLIB" "$IMG" \
    /arf/scratch/egitimg16/nedolm_0p8b/jobs/container_python.sh "$CODE/training/eval_role.py" --mix "$MIX" "$@"
}
run --role yasama --checkpoint "$YASAMA" --out "$OUT/yasama"
run --role yurutme --checkpoint "$YURUTME" --out "$OUT/yurutme"
run --role yargi --checkpoint "$YARGI" --out "$OUT/yargi" --chat
run --role yargi --checkpoint "$BASE" --out "$OUT/yargi_dpo_base" --chat
echo EVAL_OK
