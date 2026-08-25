#!/bin/bash
#SBATCH --job-name=pag-train-fir
#SBATCH --account=def-bboulet_gpu
#SBATCH --partition=gpubase_bynode_b4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Fir: 2 nodes × 4×H100 = 8 GPUs (same world size as Nibi 1×8).
# Reuses run_pag_local.sh + ray_bootstrap_narval.sh — do not copy venv from Nibi.
#
#   cd $SCRATCH/verify_then_rectify/PAG-reproduction
#   mkdir -p logs
#   sbatch quick_start/train_pag_fir.slurm
#
# Smoke (1 train step):
#   sbatch --time=1:00:00 --partition=gpubase_bynode_b1 \
#     --export=ALL,SMOKE=1,EXPERIMENT_NAME=smoke_qwen7b_fir \
#     quick_start/train_pag_fir.slurm
#
# Resume from Nibi global_step_180: keep EXPERIMENT_NAME and copy that ckpt dir
# to checkpoints/PAG/<EXPERIMENT_NAME>/ on Fir first. resume_mode=auto.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if [[ "$(basename "$REPO_ROOT")" == "quick_start" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/checkpoints"

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10.13
# shellcheck source=/dev/null
source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh
# shellcheck source=/dev/null
source "$REPO_ROOT/quick_start/ray_bootstrap_narval.sh"

export HF_HOME="${HF_HOME:-/scratch/yuranli/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/checkpoints}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
export N_GPUS=4
export NNODES=2
export SMOKE="${SMOKE:-0}"
export USE_WANDB="${USE_WANDB:-1}"
# Same name as Nibi so auto-resume finds global_step_180 after rsync.
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen7b_pag_8xh100_nibi}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node="${nodes[0]}"
port=6379
echo "[train-fir] resolving head IP via srun..."
head_ip="$(ray_resolve_head_ip "$head_node")"
echo "[train-fir] head=${head_node} ip=${head_ip} nodes=${nodes[*]}"
echo "[train-fir] MODEL_PATH=${MODEL_PATH} N_GPUS=${N_GPUS} NNODES=${NNODES} GPU_MEM_UTIL=${GPU_MEM_UTIL}"
echo "[train-fir] SMOKE=${SMOKE} EXPERIMENT_NAME=${EXPERIMENT_NAME} USE_WANDB=${USE_WANDB}"

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  bash -lc "
    set -euo pipefail
    source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh
    source '${REPO_ROOT}/quick_start/ray_bootstrap_narval.sh'
    if [[ \"\${SLURMD_NODENAME}\" == '${head_node}' ]]; then
      ray_start_head '${head_ip}' '${port}' '${N_GPUS}'
    else
      sleep 10
      ray_start_worker '${head_ip}' '${port}' '${N_GPUS}'
    fi
  "

export RAY_ADDRESS="${head_ip}:${port}"
ray_wait_ready "$RAY_ADDRESS" 60

cleanup() {
  echo '[train-fir] shutting down ray...'
  srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    bash -lc 'source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh; ray stop --force || true' \
    || true
}
trap cleanup EXIT

echo "[train-fir] RAY_ADDRESS=${RAY_ADDRESS}"
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
      MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
      MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
      ${TRAIN_FILE:+TRAIN_FILE="$TRAIN_FILE"} \
      bash "$REPO_ROOT/quick_start/run_pag_local.sh"
status=$?
set -e

if [[ $status -eq 0 ]]; then
  echo "[train-fir] PASS (exit=0)."
else
  echo "[train-fir] FAIL (exit=${status}). Check logs for CUDA OOM / Ray / vLLM / NCCL errors."
fi
exit "$status"
