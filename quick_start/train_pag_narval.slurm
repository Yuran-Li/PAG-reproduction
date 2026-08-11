#!/bin/bash
#SBATCH --job-name=pag-math7b
#SBATCH --account=def-bboulet_gpu
#SBATCH --partition=gpubase_bynode_b4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=2-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Narval: 2 nodes × 4×A100 = 8 GPUs (closest to upstream single-node 8-GPU).
#   mkdir -p logs
#   sbatch quick_start/train_pag_narval.slurm
#
# Optional:
#   sbatch --export=ALL,USE_WANDB=1,EXPERIMENT_NAME=my_run,MODEL_PATH=Qwen/Qwen2.5-Math-7B-Instruct \
#     quick_start/train_pag_narval.slurm

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
export USE_WANDB="${USE_WANDB:-0}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-math7b_pag_narval8}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node="${nodes[0]}"
port=6379

echo "[slurm] resolving head IP (hostname -i / getent)..."
head_ip=$(ray_resolve_head_ip "$head_node")
echo "[slurm] head=${head_node} ip=${head_ip} nodes=${nodes[*]}"
echo "[slurm] MODEL_PATH=${MODEL_PATH} N_GPUS=${N_GPUS} NNODES=${NNODES} GPU_MEM_UTIL=${GPU_MEM_UTIL}"
echo "[slurm] RAY_TMPDIR will be /tmp/r${SLURM_JOB_ID} (short path for AF_UNIX)"

srun --nodes=1 --ntasks=1 -w "$head_node" \
  bash -c "set -euo pipefail; source '$REPO_ROOT/quick_start/ray_bootstrap_narval.sh'; ray_start_head '${head_ip}' '${port}' '${N_GPUS}'"

export RAY_ADDRESS="${head_ip}:${port}"
ray_wait_ready "$RAY_ADDRESS" 60

worker_nodes=("${nodes[@]:1}")
for wnode in "${worker_nodes[@]}"; do
  srun --nodes=1 --ntasks=1 -w "$wnode" \
    bash -c "set -euo pipefail; source '$REPO_ROOT/quick_start/ray_bootstrap_narval.sh'; ray_start_worker '${head_ip}' '${port}' '${N_GPUS}'"
done

cleanup() {
  echo '[slurm] shutting down ray...'
  srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    bash -c 'source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh; ray stop --force || true' \
    || true
}
trap cleanup EXIT

echo "[slurm] RAY_ADDRESS=${RAY_ADDRESS}"
ray status --address="$RAY_ADDRESS" || true

srun --nodes=1 --ntasks=1 -w "$head_node" \
  env RAY_ADDRESS="$RAY_ADDRESS" \
      N_GPUS="$N_GPUS" \
      NNODES="$NNODES" \
      MODEL_PATH="$MODEL_PATH" \
      CKPT_PATH="$CKPT_PATH" \
      HF_HOME="$HF_HOME" \
      HF_HUB_CACHE="$HF_HUB_CACHE" \
      TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
      USE_WANDB="$USE_WANDB" \
      EXPERIMENT_NAME="$EXPERIMENT_NAME" \
      GPU_MEM_UTIL="$GPU_MEM_UTIL" \
      bash "$REPO_ROOT/quick_start/run_pag_local.sh"

echo '[slurm] training finished'
