#!/bin/bash
#SBATCH --job-name=pag-merge-nibi
#SBATCH --account=def-bboulet_cpu
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128000M
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Merge FSDP actor shards -> HuggingFace. Login nodes will OOM-kill this.
#   cd /scratch/yuranli/verify_then_rectify/PAG-reproduction
#   mkdir -p logs
#   sbatch quick_start/merge_ckpt_nibi.slurm
#
# Override:
#   sbatch --export=ALL,CKPT_STEP=180,EXPERIMENT_NAME=qwen7b_pag_8xh100_nibi \
#     quick_start/merge_ckpt_nibi.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if [[ "$(basename "$REPO_ROOT")" == "quick_start" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs"

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10.13
# shellcheck source=/dev/null
source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh

export HF_HOME="${HF_HOME:-/scratch/yuranli/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen7b_pag_8xh100_nibi}"
CKPT_STEP="${CKPT_STEP:-180}"
HF_MODEL_PATH="${HF_MODEL_PATH:-$HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
LOCAL_DIR="${LOCAL_DIR:-$REPO_ROOT/checkpoints/PAG/${EXPERIMENT_NAME}/global_step_${CKPT_STEP}/actor}"
TARGET_DIR="${TARGET_DIR:-$REPO_ROOT/checkpoints/PAG/${EXPERIMENT_NAME}/global_step_${CKPT_STEP}/actor_hf}"

echo "[merge] host=$(hostname) mem=128G cpus=${SLURM_CPUS_PER_TASK}"
echo "[merge] HF_MODEL_PATH=${HF_MODEL_PATH}"
echo "[merge] LOCAL_DIR=${LOCAL_DIR}"
echo "[merge] TARGET_DIR=${TARGET_DIR}"
test -f "$LOCAL_DIR/model_world_size_8_rank_0.pt"
test -f "$HF_MODEL_PATH/config.json"

python scripts/model_merger.py \
  --backend fsdp \
  --hf_model_path "$HF_MODEL_PATH" \
  --local_dir "$LOCAL_DIR" \
  --target_dir "$TARGET_DIR"

echo "[merge] done. listing ${TARGET_DIR}"
ls -lh "$TARGET_DIR" | head -30
du -sh "$TARGET_DIR"
