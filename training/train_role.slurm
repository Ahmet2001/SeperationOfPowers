#!/bin/bash
# One Mercan role on one GPU (TRUBA kolyoz-cuda, submit from cuda-ui).
#   sbatch --export=ALL,ROLE=yasama,RUN_DIR=...,CODE=...,MIX=...,EXTRA="--max-steps 30" training/train_role.slurm
#SBATCH -J mercan_role
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg16
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
set -euo pipefail
: "${ROLE:?}" "${RUN_DIR:?}" "${CODE:?}" "${MIX:?}"
IMG=/arf/sw/containers/truba-ai/pytorch-26.06-py3.sif
CACHE=/tmp/$USER/mercan-role-$SLURM_JOB_ID
mkdir -p "$RUN_DIR" "$CACHE"
export TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor CUDA_CACHE_PATH=$CACHE/nv
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
apptainer exec --nv --bind /arf:/arf --bind /tmp:/tmp \
  --env PYTHONPATH="$CODE" \
  "$IMG" /arf/scratch/egitimg16/nedolm_0p8b/jobs/container_python.sh \
  "$CODE/training/train_role.py" --role "$ROLE" --mix "$MIX" --run-dir "$RUN_DIR" ${EXTRA:-}
