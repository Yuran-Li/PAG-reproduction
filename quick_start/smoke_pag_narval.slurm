#!/bin/bash
#SBATCH --job-name=pag-smoke7b
#SBATCH --account=def-bboulet_gpu
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Automatic OOM / launch smoke test: 2×4 A100, 1 hour walltime.
# Runs val_before_train + 1 training step then exits (SMOKE=1).
#
#   cd /scratch/yuranli/verify_then_rectify/PAG-reproduction
#   mkdir -p logs
#   sbatch quick_start/smoke_pag_narval.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if [[ "$(basename "$REPO_ROOT")" == "quick_start" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/checkpoints"

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10.13
source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh
# shellcheck source=/dev/null
source "$REPO_ROOT/quick_start/ray_bootstrap_narval.sh"

export HF_HOME="${HF_HOME:-/scratch/yuranli/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/checkpoints}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Math-7B-Instruct}"
export N_GPUS=4
export NNODES=2
export SMOKE=1
export USE_WANDB="${USE_WANDB:-0}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-smoke_math7b_8xa100}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node="${nodes[0]}"
port=6379

echo "[smoke] resolving head IP (hostname -i / getent)..."
head_ip=$(ray_resolve_head_ip "$head_node")
echo "[smoke] head=${head_node} ip=${head_ip} nodes=${nodes[*]}"
echo "[smoke] MODEL_PATH=${MODEL_PATH} N_GPUS=${N_GPUS} NNODES=${NNODES} GPU_MEM_UTIL=${GPU_MEM_UTIL}"
echo "[smoke] RAY_TMPDIR will be /tmp/r${SLURM_JOB_ID} (short path for AF_UNIX)"

# 1) Start Ray head first
srun --nodes=1 --ntasks=1 -w "$head_node" \
  bash -c "set -euo pipefail; source '$REPO_ROOT/quick_start/ray_bootstrap_narval.sh'; ray_start_head '${head_ip}' '${port}' '${N_GPUS}'"

export RAY_ADDRESS="${head_ip}:${port}"
ray_wait_ready "$RAY_ADDRESS" 60

# 2) Start workers on remaining nodes
worker_nodes=("${nodes[@]:1}")
for wnode in "${worker_nodes[@]}"; do
  srun --nodes=1 --ntasks=1 -w "$wnode" \
    bash -c "set -euo pipefail; source '$REPO_ROOT/quick_start/ray_bootstrap_narval.sh'; ray_start_worker '${head_ip}' '${port}' '${N_GPUS}'"
done

cleanup() {
  echo '[smoke] shutting down ray...'
  srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    bash -c 'source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh; ray stop --force || true' \
    || true
}
trap cleanup EXIT

echo "[smoke] RAY_ADDRESS=${RAY_ADDRESS}"
ray status --address="$RAY_ADDRESS" || true

set +e
srun --nodes=1 --ntasks=1 -w "$head_node" \
  env RAY_ADDRESS="$RAY_ADDRESS" \
      N_GPUS="$N_GPUS" \
      NNODES="$NNODES" \
      SMOKE="$SMOKE" \
      MODEL_PATH="$MODEL_PATH" \
      CKPT_PATH="$CKPT_PATH" \
      HF_HOME="$HF_HOME" \
      HF_HUB_CACHE="$HF_HUB_CACHE" \
      TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
      USE_WANDB="$USE_WANDB" \
      EXPERIMENT_NAME="$EXPERIMENT_NAME" \
      GPU_MEM_UTIL="$GPU_MEM_UTIL" \
      bash "$REPO_ROOT/quick_start/run_pag_local.sh"
status=$?
set -e

if [[ $status -eq 0 ]]; then
  echo "[smoke] PASS (exit=0). Likely no OOM on launch/1-step."
else
  echo "[smoke] FAIL (exit=${status}). Check logs for CUDA OOM / Ray errors."
fi
exit "$status"
